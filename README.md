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
|3x3 Matrix|<img width="2959" height="1974" alt="C_out_matrix" src="https://github.com/user-attachments/assets/832f1436-b92c-4451-bfa8-377cd8ebe4e1" />|<img width="2959" height="1974" alt="C_out_matrix_Y" src="https://github.com/user-attachments/assets/d7bfb3bc-85cd-49f2-a658-e1a309afebc1" />|
|RPCC4|<img width="2959" height="1974" alt="C_out_rpcc" src="https://github.com/user-attachments/assets/0e34d2be-a2d6-480c-94dc-8edeb27654e4" />|<img width="2959" height="1974" alt="C_out_rpcc_Y_rb20" src="https://github.com/user-attachments/assets/3a4b54ac-9930-48ac-9d1b-46caed87380c" />|


