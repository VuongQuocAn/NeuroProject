ALTER TABLE analysis_results
    ADD COLUMN IF NOT EXISTS finer_cam_path VARCHAR,
    ADD COLUMN IF NOT EXISTS seg_eigen_cam_path VARCHAR,
    ADD COLUMN IF NOT EXISTS odam_path VARCHAR,
    ADD COLUMN IF NOT EXISTS xai_3_panel_path VARCHAR,
    ADD COLUMN IF NOT EXISTS survival_curve_data JSON;
