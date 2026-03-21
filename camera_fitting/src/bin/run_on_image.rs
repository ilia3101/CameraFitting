// #![allow(dead_code, unused_variables, unused_imports, unused_mut, unused_parens)]

use camera_fitting::{
    CameraTransform, MatrixTransform,
    root_polynomial::{Rpcc2, Rpcc3, Rpcc4},
    utility_transforms::Gained,
};
use colour::spectrum::{DiscreteSpectrum, GaussianSpectrum, Spectrum, XySpectrum};
use maths::linear_algebra::{Matrix, Vector};

use image_lib::{ImageBuffer, Pixel, Rgb};

/* Loads a raw image, demosaicing by doing a 2x downscale */
fn load_image(path: &str) -> (usize, usize, Vec<f32>) {
    #[cfg(feature = "raw_demo")]
    {
        let image = rawloader::decode_file(path).unwrap();
        if let rawloader::RawImageData::Integer(data) = image.data {
            let wl = image.whitelevels[0] as f32;
            let bl = image.blacklevels[0] as f32;
            let width = image.width;
            let height = image.height;
            let width_final = image.width / 2 - 1;
            let height_final = image.height / 2 - 1;

            // if image.cfa.name == "RGGB"; //TODO: use offsets for non RGGB cfa patterns
            let mut as_rgb = vec![0.; width_final * height_final * 3];

            let to_range = |x: u16| (x as f32 - bl) / (wl - bl);
            for y in 0..height_final {
                for x in 0..width_final {
                    let out_off = (y * width_final + x) * 3;
                    let mut rgb = Vector([
                        to_range(data[(y * 2) * width + x * 2]),
                        (to_range(data[(y * 2) * width + x * 2 + 1])
                            + to_range(data[(y * 2 + 1) * width + x * 2]))
                            / 2.0,
                        to_range(data[(y * 2 + 1) * width + x * 2 + 1]),
                    ]);
                    // fix channels with below zero values to help with noise
                    if rgb.min() < 0.0 {
                        let (min, max) = (rgb.min(), rgb.max());
                        rgb = rgb.map(|x| (x - max) * (max / (max - min)) + max);
                    }
                    as_rgb[out_off + 0] = rgb[0];
                    as_rgb[out_off + 1] = rgb[1];
                    as_rgb[out_off + 2] = rgb[2];
                }
            }

            // apply a blurring filter to smooth noise effects on colour
            let mut blurred = vec![0.; width_final * height_final * 3];
            for y in 1..(height_final - 1) {
                for x in 1..(width_final - 1) {
                    let off_prev = ((y - 1) * width_final + x) * 3;
                    let off = (y * width_final + x) * 3;
                    let off_next = ((y + 1) * width_final + x) * 3;
                    for c in 0..3 {
                        blurred[off + c] = (as_rgb[off + c] * 2.0
                            + as_rgb[off + 3 + c]
                            + as_rgb[off - 3 + c]
                            + as_rgb[off_prev + c]
                            + as_rgb[off_next + c])
                            / 6.0;
                    }
                }
            }

            return (width_final, height_final, blurred);
        }
    }
    panic!("Run with feature raw_demo enabled for this binary")
}

#[derive(Copy, Clone, Debug)]
struct ProcessingParams {
    exposure: f32,
    contrast: f32,
    middle_grey: f32,
    inset: f32,
    wb: Vector<f32, 3>,
}

fn process_and_save(
    w: usize,
    h: usize,
    mut data: Vec<f32>,
    process: ProcessingParams,
    camera_profile: impl CameraTransform<f32>,
    out: &str,
) {
    let to_srgb = (Matrix([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]))
    .as_f32();

    let inset = Matrix([
        [
            1.0 - process.inset,
            process.inset / 2.0,
            process.inset / 2.0,
        ],
        [
            process.inset * 0.3,
            1.0 - process.inset,
            process.inset * 0.7,
        ],
        [
            process.inset / 2.0,
            process.inset / 2.0,
            1.0 - process.inset,
        ],
    ]);

    for pix in data.as_chunks_mut::<3>().0.iter_mut() {
        let rgb = Vector(*pix);
        // apply profile and do basic white balance
        let calibrated = camera_profile.apply(rgb) * process.wb;

        // convert to srgb
        let as_srgb = to_srgb * calibrated;

        // clip and apply agx-ish inset
        let inset = inset * (as_srgb).map(|x| x.max(0.));

        // apply look processing per channel
        let processed = inset.map(|x| {
            let with_contrast = ((x * process.exposure) / process.middle_grey)
                .powf(process.contrast)
                * process.middle_grey;
            // apply basic tonemapping curve
            let tonemapped = with_contrast / (1.0 + with_contrast);
            tonemapped
        });

        *pix = processed.0;
    }

    // Save it
    let image = ImageBuffer::<Rgb<u8>, _>::from_vec(
        w as u32,
        h as u32,
        data.iter().map(|x| fast_srgb8::f32_to_srgb8(*x)).collect(),
    )
    .unwrap();
    image.save(out).unwrap();
}

fn process_and_save_luminance(
    w: usize,
    h: usize,
    mut data: Vec<f32>,
    process: ProcessingParams,
    camera_profile: impl CameraTransform<f32>,
    out: &str,
) {
    for pix in data.as_chunks_mut::<3>().0.iter_mut() {
        let rgb = Vector(*pix);
        // apply profile and do basic white balance
        let calibrated = camera_profile.apply(rgb);

        // apply look processing per channel
        let processed = Vector([calibrated[1], calibrated[1], calibrated[1]]).map(|x| {
            let with_contrast = ((x * process.exposure) / process.middle_grey)
                .powf(process.contrast)
                * process.middle_grey;
            // apply basic tonemapping curve
            let tonemapped = with_contrast / (1.0 + with_contrast);
            tonemapped
        });

        *pix = processed.0;
    }

    // Save it
    let image = ImageBuffer::<Rgb<u8>, _>::from_vec(
        w as u32,
        h as u32,
        data.iter().map(|x| fast_srgb8::f32_to_srgb8(*x)).collect(),
    )
    .unwrap();
    image.save(out).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <image_path> <camera_profile>", args[0]);
        std::process::exit(1);
    }

    let image_path = &args[1];
    let camera_profile_rpcc = Gained {
        trans: Rpcc4(
            Vector([
                -11.211099669013414,
                -3.4139176036743635,
                10.745832346111243,
                617.4894699514323,
                -417.1184445828426,
                10.896150008547632,
                -634.3889280082111,
                444.70810240505875,
                -7.92189388465737,
                -612.5357855246349,
                420.9285972944314,
                -10.243699534329565,
                4.236150970197573,
                318.1014628989084,
                5.40776795155279,
                336.69008211538136,
                -227.0919199263956,
                3.7839829469381447,
                -242.50727634710015,
                -3.5444751167777673,
                1.979595923605648,
                -4.420385739674456,
            ]),
            Vector([
                -7.044399235551847,
                -4.437609567043381,
                3.147588942471849,
                377.142240543759,
                -148.7796639706878,
                5.244201827639518,
                -393.7146448084469,
                153.61042468764418,
                -2.810150265829772,
                -372.29361832037785,
                151.36150874649124,
                -5.01895086310277,
                5.027674700969887,
                193.69851533447746,
                2.7175272752941013,
                211.1038664328387,
                -82.48201788054749,
                1.0626024146800797,
                -81.06463505934306,
                -3.24773491759292,
                1.4593188913919253,
                -4.089444430568146,
            ]),
            Vector([
                -16.20529227193962,
                39.56331667608808,
                61.044555636164816,
                789.1774470041407,
                -2688.670323093358,
                -2.8156247159477097,
                -819.785937588838,
                2754.611202915936,
                16.32369989466132,
                -794.7757969803293,
                2717.087626448726,
                -2.303477458050084,
                -35.23254843315562,
                418.4664445187351,
                -0.3495994526868225,
                440.42900421922053,
                -1439.4062469935836,
                -16.420998373238888,
                -1464.2819073346104,
                13.641392991468349,
                13.007272487136207,
                17.35937185014341,
            ]),
        ),
        gain: Vector([2.452324311364167, 1.269823923246411, 2.1639891839458754]),
    };

    let camera_profile_mat = MatrixTransform {
        matrix: Matrix([
            [1.187070896745104, 0.021995847574612853, 0.1446606494873752],
            [0.4507521191285102, 0.6087915340471513, -0.15246386272924792],
            [0.09635250386959734, -0.3127749600064613, 1.4466109103141598],
        ]),
    };

    let (w, h, data) = load_image(image_path);

    let process_pars = ProcessingParams {
        exposure: 4.2,
        contrast: 1.3,
        middle_grey: 0.18,
        inset: 0.16,
        wb: Vector([1.0, 1.03, 1.25]),
    };

    process_and_save(
        w,
        h,
        data.clone(),
        process_pars,
        camera_profile_rpcc,
        "out_rpcc.png",
    );
    process_and_save(
        w,
        h,
        data.clone(),
        process_pars,
        camera_profile_mat,
        "out_matrix.png",
    );
    process_and_save_luminance(
        w,
        h,
        data.clone(),
        process_pars,
        camera_profile_rpcc,
        "out_rpcc_Y.png",
    );
    process_and_save_luminance(
        w,
        h,
        data.clone(),
        process_pars,
        camera_profile_mat,
        "out_matrix_Y.png",
    );
}
