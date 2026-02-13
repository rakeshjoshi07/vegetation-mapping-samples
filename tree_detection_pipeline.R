#!/usr/bin/env Rscript

# Individual Tree Detection Pipeline for Continental Australia
# Processes LiDAR tiles using optimized CHM and segmentation methods
# Scales to 200-node HPC parallel processing

Sys.setlocale("LC_ALL", "C.UTF-8")

Sys.setenv(
  OGR_SQLITE_SYNCHRONOUS = "OFF",
  OGR_SQLITE_CACHE = "512",
  OGR_GPKG_USE_RTREE = "NO"
)

personal_lib <- "/datasets/work/ev-anu-nasa/work/MiniOzTree/r_library"
home_lib <- path.expand("~/R_libs")
.libPaths(c(home_lib, personal_lib, .libPaths()))

options(repos = "https://cloud.r-project.org")

suppressPackageStartupMessages({
  library(sf)
  library(fs)
  library(dplyr)
  library(data.table)
  library(lidR)
  library(terra)
  library(stars)
})

options(lidR.progress = FALSE)


# =============================================================================
# Configuration
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
node_id <- as.numeric(args[1])
n_nodes <- 200

pipeline_df_path <- "/datasets/work/ev-anu-nasa/work/MiniOzTree/Rakesh_processing/minioztree_pipeline.csv"
best_delineations_path <- "/datasets/work/ev-anu-nasa/work/MiniOzTree/Rakesh_processing/best_delineations.csv"

output_path_chms <- "/datasets/work/ev-anu-nasa/work/MiniOzTree/Rakesh_processing/chms"
output_path_ttops <- "/datasets/work/ev-anu-nasa/work/MiniOzTree/Rakesh_processing/ttops"
output_path_crowns <- "/datasets/work/ev-anu-nasa/work/MiniOzTree/Rakesh_processing/crowns"
output_path_footprints <- "/datasets/work/ev-anu-nasa/work/MiniOzTree/Rakesh_processing/footprints"
output_path_metadata <- "/datasets/work/ev-anu-nasa/work/MiniOzTree/Rakesh_processing/processing_metadata"

dir_create(c(output_path_chms, output_path_ttops, output_path_crowns, 
             output_path_footprints, output_path_metadata))


# =============================================================================
# Helper Functions
# =============================================================================

split_files <- function(n_nodes, n_files) {
  files_per_node <- ceiling(n_files / n_nodes)
  splits <- vector("list", n_nodes)
  for (i in seq_len(n_nodes)) {
    start <- (i - 1) * files_per_node + 1
    end <- min(i * files_per_node, n_files)
    splits[[i]] <- c(start, end)
  }
  splits
}

density_class_from_pts <- function(pts_m2) {
  if (pts_m2 < 3) return("low")
  if (pts_m2 < 16) return("medium")
  "high"
}

create_f <- function(intercept, slope, se, mini, maxi, q25, q75) {
  force(intercept); force(slope); force(se)
  force(mini); force(maxi); force(q25); force(q75)
  function(x) {
    y <- (intercept + slope * x + se) * 2
    y[x < mini] <- q25
    y[x > maxi] <- q75
    y
  }
}


# =============================================================================
# Load Data and Split Work
# =============================================================================

pipeline_df <- fread(pipeline_df_path)
best_delineations <- fread(best_delineations_path, header = TRUE, stringsAsFactors = FALSE)

splits <- split_files(n_nodes, nrow(pipeline_df))
start_idx <- splits[[node_id]][1]
end_idx <- splits[[node_id]][2]
inputs <- pipeline_df[start_idx:end_idx, ]

cat("\n============================================\n")
cat(" MiniOzTree processing started\n")
cat(" Node ID:", node_id, "\n")
cat(" Tiles assigned:", start_idx, "to", end_idx, "\n")
cat(" Tiles to process:", nrow(inputs), "\n")
cat("============================================\n")


# =============================================================================
# Main Processing Loop
# =============================================================================

metadata_df <- list()
skipped_count <- 0
processed_count <- 0
failed_count <- 0

for (i in seq_len(nrow(inputs))) {
  row <- inputs[i, ]
  
  laz_path <- row$laz_path
  vc <- row$vc_name
  epsg <- row$epsg
  tile_name <- tools::file_path_sans_ext(basename(laz_path))
  
  chm_exists <- file.exists(file.path(output_path_chms, paste0(tile_name, "_chm.tif")))
  crowns_exists <- file.exists(file.path(output_path_crowns, paste0(tile_name, "_crowns.gpkg")))
  
  if (chm_exists && crowns_exists) {
    cat("\n--------------------------------------------\n")
    cat("Tile:", tile_name, "- ALREADY PROCESSED, SKIPPING\n")
    cat("--------------------------------------------\n")
    skipped_count <- skipped_count + 1
    next
  }
  
  cat("\n--------------------------------------------\n")
  cat("Tile:", tile_name, "\n")
  cat("Vegetation class:", vc, "\n")
  cat("EPSG:", epsg, "\n")
  cat("--------------------------------------------\n")
  
  if (!file.exists(laz_path)) {
    cat("✗ File missing, skipping\n")
    failed_count <- failed_count + 1
    next
  }
  
  las <- tryCatch({
    readLAS(laz_path)
  }, error = function(e) {
    cat("✗ Error reading LAS:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(las) || is.empty(las)) {
    cat("✗ Empty LAS, skipping\n")
    failed_count <- failed_count + 1
    next
  }
  
  las <- las_rescale(las, 0.001, 0.001, 0.001)
  st_crs(las) <- epsg
  
  area_tile <- area(las)
  n_fr <- las@header$`Number of points by return`[1]
  pts_m2 <- n_fr / area_tile
  dens_cls <- density_class_from_pts(pts_m2)
  
  cat("✓ Density:", round(pts_m2, 2), "pts/m² (", dens_cls, ")\n")
  
  sel <- best_delineations %>%
    filter(veg_class == vc, density_class == dens_cls)
  
  if (nrow(sel) == 0) {
    cat("✗ No delineation match\n")
    failed_count <- failed_count + 1
    next
  }
  
  canopy_algo_name <- tolower(sel$chm_meth[1])
  segmentation_algo_name <- tolower(sel$its_meth[1])
  
  cat("✓ CHM:", canopy_algo_name, " | ITS:", segmentation_algo_name, "\n")
  
  f <- create_f(
    intercept = row$itd_intercept,
    slope = row$itd_slope,
    se = row$itd_se,
    mini = row$itd_mini,
    maxi = row$itd_maxi,
    q25 = row$itd_q25,
    q75 = row$itd_q75
  )
  
  nlas <- tryCatch({
    normalize_height(las, knnidw())
  }, error = function(e) {
    cat("✗ Normalization failed:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(nlas)) {
    failed_count <- failed_count + 1
    next
  }
  
  nlas@data <- nlas@data[!(Classification %in% c(3,4,5)), Z := 0]
  
  ttops <- tryCatch({
    locate_trees(nlas, lmf(f))
  }, error = function(e) {
    cat("✗ Tree detection failed:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(ttops) || nrow(ttops) == 0) {
    cat("✗ No treetops detected\n")
    failed_count <- failed_count + 1
    next
  }
  
  if (nrow(ttops) < 5) {
    cat("✗ Too few treetops (", nrow(ttops), "), skipping\n")
    failed_count <- failed_count + 1
    next
  }
  
  cat("✓ Treetops:", nrow(ttops), "\n")
  
  tryCatch({
    st_write(ttops, file.path(output_path_ttops, paste0(tile_name, "_ttops.gpkg")),
             delete_dsn = TRUE, quiet = TRUE)
  }, error = function(e) {
    cat("⚠ Warning: Could not save treetops:", e$message, "\n")
  })
  
  chm_algo <- switch(canopy_algo_name,
    pit = pitfree(),
    p2r = p2r(subcircle = 0.2),
    tin = dsmtin(),
    stop("Unknown CHM method")
  )
  
  chm <- tryCatch({
    rasterize_canopy(nlas, res = 0.5, chm_algo)
  }, error = function(e) {
    cat("✗ CHM generation failed:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(chm)) {
    failed_count <- failed_count + 1
    next
  }
  
  chm_saved <- tryCatch({
    terra::writeRaster(chm, file.path(output_path_chms, paste0(tile_name, "_chm.tif")),
                      overwrite = TRUE)
    TRUE
  }, error = function(e) {
    cat("✗ CHM save failed:", e$message, "\n")
    FALSE
  })
  
  if (!chm_saved) {
    failed_count <- failed_count + 1
    next
  }
  
  cat("✓ CHM saved\n")
  
  seg_algo <- tryCatch(
    switch(segmentation_algo_name,
      dalponte = dalponte2016(chm, ttops),
      watershed = lidR::watershed(chm),
      li = li2012(),
      stop("Unknown ITS method")
    ),
    error = function(e) {
      cat("✗ ITS algorithm creation failed:", e$message, "\n")
      return(NULL)
    }
  )
  
  if (is.null(seg_algo)) {
    failed_count <- failed_count + 1
    next
  }
  
  nlas <- tryCatch({
    segment_trees(nlas, seg_algo)
  }, error = function(e) {
    cat("✗ Segmentation failed:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(nlas)) {
    failed_count <- failed_count + 1
    next
  }
  
  if (!"treeID" %in% names(nlas@data)) {
    cat("✗ No treeID column - segmentation failed\n")
    failed_count <- failed_count + 1
    next
  }
  
  n_trees_segmented <- length(unique(nlas@data$treeID[!is.na(nlas@data$treeID)]))
  
  if (n_trees_segmented == 0) {
    cat("✗ No trees segmented\n")
    failed_count <- failed_count + 1
    next
  }
  
  cat("✓ Trees segmented:", n_trees_segmented, "\n")
  
  crowns <- tryCatch({
    crown_metrics(nlas, .stdtreemetrics, geom = "convex")
  }, error = function(e) {
    cat("✗ Crown metrics failed:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(crowns)) {
    cat("✗ Crown metrics returned NULL\n")
    failed_count <- failed_count + 1
    next
  }
  
  if (nrow(crowns) == 0) {
    cat("✗ No valid crowns generated\n")
    failed_count <- failed_count + 1
    next
  }
  
  cat("✓ Crowns:", nrow(crowns), "\n")
  
  crowns_saved <- tryCatch({
    st_write(crowns, file.path(output_path_crowns, paste0(tile_name, "_crowns.gpkg")),
             delete_dsn = TRUE, quiet = TRUE)
    TRUE
  }, error = function(e) {
    cat("✗ Crown save failed:", e$message, "\n")
    FALSE
  })
  
  if (!crowns_saved) {
    failed_count <- failed_count + 1
    next
  }
  
  cat("✓ Processing complete\n")
  
  metadata_df[[length(metadata_df) + 1]] <- data.frame(
    tile_name = tile_name,
    vc = vc,
    density_class = dens_cls,
    chm = canopy_algo_name,
    its = segmentation_algo_name,
    pts_m2 = pts_m2,
    crowns = nrow(crowns),
    stringsAsFactors = FALSE
  )
  
  processed_count <- processed_count + 1
  gc()
}


# =============================================================================
# Save Results
# =============================================================================

if (length(metadata_df) > 0) {
  metadata_df <- bind_rows(metadata_df)
  metadata_out <- file.path(output_path_metadata,
                           paste0("processing_metadata_node_", node_id, ".csv"))
  write.csv(metadata_df, metadata_out, row.names = FALSE)
  
  cat("\n============================================\n")
  cat(" Node", node_id, "completed successfully\n")
  cat(" Skipped (already processed):", skipped_count, "tiles\n")
  cat(" Newly processed:", processed_count, "tiles\n")
  cat(" Failed:", failed_count, "tiles\n")
  cat("============================================\n")
} else {
  cat("\n============================================\n")
  cat(" Node", node_id, "completed\n")
  cat(" Skipped (already processed):", skipped_count, "tiles\n")
  cat(" Failed:", failed_count, "tiles\n")
  cat(" No new tiles successfully processed\n")
  cat("============================================\n")
}
