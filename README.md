# CameraFitting
Experimenting with camera colour calibration methods, including RPCC (root polynomial colour transform)

- Main code for camera fitting is in camera_fitting/src/bin/camera_fitting.rs
- Spectral data from my [SpectralMeasurements](https://github.com/ilia3101/spectralmeasurements) camera spectral response measurement repo
- Non linear least squares optimiser taken from my [Panorama](https://github.com/ilia3101/Panorama) repo

Example command to run fitting:
```
cargo run --release --bin camera_fitting  "colour/spectral_data/old_camera_measurements_by_me/Canon/EOS 5D Mark III/001/response.dat"
```

## Purpose

- Improve hue correctness over poor 3x3 matrix profiles
- Fix negative luminance/extreme values

## Example

Using the run_on_image binary, I generated the following outputs. The difference is not immediately apparent in the processed image due to AgX processing which handles strong colours effecively, but if you look closely you will see much smoother colour transitions on the RPCC image, particularly where shadows and colours intersect. The difference is most clear in the Luminance channel of course!

|profile type|Processed image (AgX)|Luminance channel (Y)|
|-|-|-|
|3x3 Matrix|<img width="2959" height="1974" alt="C_out_matrix" src="https://github.com/user-attachments/assets/7c6e2a71-0328-40d8-af9d-755cc670c077" />|<img width="2959" height="1974" alt="C_out_matrix_Y" src="https://github.com/user-attachments/assets/6f896ab2-b956-48f0-9b52-cbece166d636" />|
|RPCC4|<img width="2959" height="1974" alt="C_out_rpcc" src="https://github.com/user-attachments/assets/d20368b0-ff7c-4fd5-a041-307d243f7a3f" />|<img width="2959" height="1974" alt="C_out_rpcc_Y_rb20" src="https://github.com/user-attachments/assets/0bee9577-48a3-4685-ba3c-f7b6e42767ca" />|
