# CameraFitting
Experimenting with camera colour calibration methods, including RPCC (root polynomial colour transform)


Main code for camera fitting is in camera_fitting/src/bin/camera_fitting.rs


Example command to run root polynomial fitting:
```
cargo run --release --bin camera_fitting  "colour/spectral_data/old_camera_measurements_by_me/Canon/EOS 5D Mark III/001/response.dat"
```
