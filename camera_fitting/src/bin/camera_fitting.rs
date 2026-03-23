#![allow(dead_code, unused_variables, unused_imports, unused_mut, unused_parens)]

use camera_fitting::{
    CameraTransform, MatrixTransform,
    root_polynomial::{Rpcc2, Rpcc3, Rpcc4},
    utility_transforms::Gained,
};
use colour::spectrum::{DiscreteSpectrum, GaussianSpectrum, Spectrum, XySpectrum};
use maths::linear_algebra::{Matrix3x3, Vector};
use maths::traits::Float;
use optimisation::{
    mappable::Mappable,
    optimise::{Optimise, OptimiseAutodiff},
    parameters::{Parametric, ToParameters},
    traits::CalculateResiduals,
};
use std::fs;
use std::fs::read_to_string;
use std::marker::PhantomData;
use utils::file::{get_cols, load_file_split_lines_map, split_lines_map};

/* Define some types */
type FloatType = f64;

/* Spectrum: discrete 2nm steps from 360-760 */
type SpectrumT<T> = DiscreteSpectrum<T, 200, 360, 2>;

#[derive(Debug, Clone, Copy)]
struct Camera<T> {
    r: SpectrumT<T>,
    g: SpectrumT<T>,
    b: SpectrumT<T>,
}

impl<T: Float> Camera<T> {
    fn integrate(&self, spectrum: SpectrumT<T>) -> Vector<T, 3> {
        let r = self.r * spectrum;
        let g = self.g * spectrum;
        let b = self.b * spectrum;
        Vector([r.sum(), g.sum(), b.sum()])
    }

    fn print_peak_values(&self)
    where
        T: std::fmt::Debug,
    {
        let (r_wl, r_val) = self.r.peak().unwrap();
        let (g_wl, g_val) = self.g.peak().unwrap();
        let (b_wl, b_val) = self.b.peak().unwrap();
        println!("R peak: wavelength={:?}nm, value={:?}", r_wl, r_val);
        println!("G peak: wavelength={:?}nm, value={:?}", g_wl, g_val);
        println!("B peak: wavelength={:?}nm, value={:?}", b_wl, b_val);
    }
}

fn parse_num(x: &str) -> f64 {
    // println!("Parsing  \"{x}\"");
    x.parse::<f64>()
        .unwrap_or_else(|_| x.parse::<i32>().unwrap_or(0) as f64)
}

fn get_camera_camspec(name: &str) -> Camera<f64> {
    let lines = include_str!("../../../colour/spectral_data/camspec_database.txt")
        .lines()
        .collect::<Vec<_>>();
    let line = lines
        .iter()
        .position(|line| line.starts_with(name))
        .unwrap();
    let mut r = lines[line + 1].split("\t").map(parse_num);
    let mut g = lines[line + 2].split("\t").map(parse_num);
    let mut b = lines[line + 3].split("\t").map(parse_num);
    let rspec = DiscreteSpectrum::<f64, 33, 400, 10>::from_fn(|_| r.next().unwrap_or(0.0));
    let gspec = DiscreteSpectrum::<f64, 33, 400, 10>::from_fn(|_| g.next().unwrap_or(0.0));
    let bspec = DiscreteSpectrum::<f64, 33, 400, 10>::from_fn(|_| b.next().unwrap_or(0.0));
    Camera {
        r: DiscreteSpectrum::from_fn(|wl| rspec.get(wl)),
        g: DiscreteSpectrum::from_fn(|wl| gspec.get(wl)),
        b: DiscreteSpectrum::from_fn(|wl| bspec.get(wl)),
    }
}

fn load_response(path: &str, separator: &str) -> Camera<f64> {
    let data: Vec<Vec<_>> = load_file_split_lines_map(path, separator, parse_num).unwrap();
    // println!("{:?}", data);
    let [wavelengths, r, g, b] = get_cols(&data, [0, 1, 2, 3]);
    let expect_msg = "Number of wavelengths and samples should match";
    Camera {
        r: XySpectrum::new(wavelengths.clone(), r)
            .expect(expect_msg)
            .discretise(),
        g: XySpectrum::new(wavelengths.clone(), g)
            .expect(expect_msg)
            .discretise(),
        b: XySpectrum::new(wavelengths.clone(), b)
            .expect(expect_msg)
            .discretise(),
    }
}

fn load_spectrum_iso(path: &str, sep: &str) -> SpectrumT<f64> {
    let data: Vec<Vec<_>> = load_file_split_lines_map(path, sep, parse_num).unwrap();
    let wavelengths: Vec<_> = data.iter().map(|row| row[0]).collect();
    let y = data.iter().map(|row| row[1]).collect();
    XySpectrum::new(wavelengths, y)
        .expect("Should work")
        .discretise()
}

fn normalise_spectra(data: &mut [SpectrumT<f64>], to_area: f64) -> &mut [SpectrumT<f64>] {
    let mut sum_area = 0f64;
    for spectrum in data.iter() {
        sum_area += spectrum.sum();
    }
    let mul = to_area / (sum_area / data.len() as f64);
    for spectrum in data.iter_mut() {
        *spectrum = *spectrum * mul;
    }
    return data;
}

fn load_iso_refl(base_path: &str) -> Vec<SpectrumT<f64>> {
    let num_spectra = 14079;
    let mut results = vec![];
    for i in 0..=num_spectra {
        let path = format!("{base_path}/{:06}.dat", i);
        results.push(load_spectrum_iso(&path, " ").normalise().unwrap());
    }
    normalise_spectra(&mut results, 1.0);
    println!("Loaded {} ISO spectra!", results.len());
    results
}

fn load_spotread_spectra(path: &str) -> Vec<SpectrumT<f64>> {
    let data: Vec<Vec<_>> = load_file_split_lines_map(path, "\t", parse_num).unwrap();
    let wavelengths = data[0][7..].to_vec();
    let mut results = vec![];
    for row in data.iter().skip(1) {
        let data = row[7..].to_vec();
        results.push(
            XySpectrum::new(wavelengths.clone(), data)
                .unwrap()
                .discretise(),
        );
    }
    normalise_spectra(&mut results, 1.0);
    println!("Loaded {} my spectra!", results.len());
    results
}

#[derive(Debug, Copy, Clone)]
struct InOutPair<T> {
    input: Vector<T, 3>,
    target: Vector<T, 3>, // lms
}

impl<A> Mappable<A> for InOutPair<A> {
    type Wrapped<B> = InOutPair<B>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> InOutPair<B>
    where
        F: FnMut(A) -> B,
    {
        InOutPair {
            input: self.input.fmap(&mut f),
            target: self.target.fmap(&mut f),
        }
    }
}

#[derive(Debug, Default, Copy, Clone)]
struct TransformOptimiser<CamTransform> {
    transform: CamTransform,
}

impl<A, CamTransform: Mappable<A>> Mappable<A> for TransformOptimiser<CamTransform> {
    type Wrapped<B> = TransformOptimiser<CamTransform::Wrapped<B>>;
    #[inline]
    fn fmap<F, B>(self, mut f: F) -> Self::Wrapped<B>
    where
        F: FnMut(A) -> B,
    {
        TransformOptimiser {
            transform: self.transform.fmap(f),
        }
    }
}

impl<T, CamTransform> CalculateResiduals<T, 3> for TransformOptimiser<CamTransform>
where
    T: Float,
    CamTransform: CameraTransform<T>,
{
    type Input = InOutPair<T>;

    type Context = ();
    fn prepare(&self) -> () {
        ()
    }

    #[inline]
    fn run(&self, ctx: &(), input: InOutPair<T>) -> [T; 3] {
        // TODO: implement more advanced residuals (some IPT/Lab difference type shit)
        (self.transform.apply(input.input) - input.target).0
    }
}

const SPECTRAL_MEASUREMENTS_PATH: &'static str = "/Users/ilia/Sync/SpectralMeasurements";

// sums all energy in all spectra in given slice
fn total_energy(data: &[SpectrumT<f64>]) -> f64 {
    let mut sum = 0.0;
    for &spectrum in data {
        sum += spectrum.sum()
    }
    sum
}

fn generate_colours(camera: Camera<f64>, illuminant: SpectrumT<f64>) -> Vec<Vector<f64, 3>> {
    // TODO: Add weighting, maybe by making everything a tuple with a weight parameter. For now,
    // weighting is implemented using multipliers as minimsation is done in linear space anyway.

    /* Weight multipliers */
    let iso_weight = 1.0;
    let my_spectra_weight = 1.0;
    let locus_weight = 1.0;
    let redblue_weight = 30.0 / 10.0;

    /************* Load reflectance datasets *************/

    /* Load ISO reflectance data */
    let mut iso_refl = load_iso_refl(&format!(
        "{SPECTRAL_MEASUREMENTS_PATH}/Data/Spectra/ReflectanceISO"
    ));

    /* Load my own data (if it's there) */
    let mut my_reflectances = vec![];
    if let Ok(dir) = fs::read_dir(
        "/Users/ilia/Sync/learning-rust/colour/spectral_data/my_reflectance_measurements/",
    ) {
        for p in dir.filter_map(|e| {
            e.ok()
                .filter(|p| p.path().extension().map(|e| e == "txt").unwrap_or(false))
        }) {
            my_reflectances.append(&mut load_spotread_spectra(&p.path().to_string_lossy()));
        }
    }

    /******** Load/generate non-reflectance spectra (emission) *********/

    // /* 1. Locus spectra */
    // let locus_min = 433.0;
    // let locus_max = 644.0;
    // let locus_steps = 212;
    // let locus_weight = 212.0 * 3.0;
    // let mut locus_spectra = vec![];
    // let locus_fwhm_bands = [10.0, 20.0, 50.0];

    // for i in 0..locus_steps {
    //     let a = i as f64 / (locus_steps - 1) as f64;
    //     let wl = locus_min + (locus_max - locus_min) * a;
    //     let area = locus_weight / (locus_steps * locus_fwhm_bands.len()) as f64;
    //     for fwhm in locus_fwhm_bands {
    //         locus_spectra.push(GaussianSpectrum { cwl: wl, fwhm: 10.0 }.discretise().normalise_area(1.0))
    //     }
    // }

    // normalise_spectra(&mut locus_spectra, 0.25);

    let mut locus_spectra = vec![];
    // add spectral locus data to keep the locus in check when optimising, with different levels of gaussian shapes...
    let mut sharp_spectral = (433..645)
        .map(|wl| {
            GaussianSpectrum {
                cwl: wl as f64,
                fwhm: 10.0,
            }
            .discretise()
            .normalise_area(1.0)
        })
        .collect::<Vec<_>>();
    let mut medium_spectral = (433..645)
        .map(|wl| {
            GaussianSpectrum {
                cwl: wl as f64,
                fwhm: 20.0,
            }
            .discretise()
            .normalise_area(1.0)
        })
        .collect::<Vec<_>>();
    let mut soft_spectral = (433..645)
        .map(|wl| {
            GaussianSpectrum {
                cwl: wl as f64,
                fwhm: 50.0,
            }
            .discretise()
            .normalise_area(1.0)
        })
        .collect::<Vec<_>>();

    normalise_spectra(&mut sharp_spectral, 0.25);
    normalise_spectra(&mut medium_spectral, 0.25);
    normalise_spectra(&mut soft_spectral, 0.25);

    locus_spectra.append(&mut sharp_spectral);
    locus_spectra.append(&mut medium_spectral);
    locus_spectra.append(&mut soft_spectral);

    /* 2. [blue-red]->white gradients, for smooth blue/red LED shadow and light transition rendering */

    let led_spectrum = |cwl: f64, fwhm: f64| -> SpectrumT<f64> {
        GaussianSpectrum { cwl, fwhm }
            .discretise()
            .normalise_area(1.0)
    };
    // TODO: use actual spectra from real led measurements
    let led_fwhm = 12.0;
    let led_pairs = [(465.0, 630.0)];
    let led_pairs_spectra = led_pairs
        .iter()
        .map(|&(a, b)| (led_spectrum(a, led_fwhm), led_spectrum(b, led_fwhm)))
        .collect::<Vec<_>>();
    // let led_pairs_spectra = [(
    //     load_spotread_spectra("/Users/ilia/Sync/learning-rust/colour/spectral_data/leds/ulanzi_rgb/blue.tsv")[0]
    //         .normalise_area(1.5),
    //     load_spotread_spectra("/Users/ilia/Sync/learning-rust/colour/spectral_data/leds/ulanzi_rgb/red.tsv")[0]
    //         .normalise_area(0.5),
    // )];
    let led_pairs_integrated = led_pairs_spectra
        .iter()
        .map(|&(a, b)| (camera.integrate(a), camera.integrate(b)))
        .collect::<Vec<_>>();
    let w_integrated = camera.integrate(illuminant.normalise_area(1.0));
    let colour_steps = 150;
    let steps_to_white = 80;
    let max_white = 0.4;

    let mut gradient_rb = vec![];

    for rb_a in (0..colour_steps).map(|i| (i as f64 / (colour_steps - 1) as f64)) {
        for w_a in (0..steps_to_white).map(|i| (i as f64 / (steps_to_white - 1) as f64)) {
            for &(r, b) in led_pairs_integrated.iter() {
                gradient_rb.push(
                    (w_integrated * w_a + (r * rb_a + b * (1.0 - rb_a)) * (1.0 - w_a))
                        * redblue_weight,
                )
            }
        }
    }

    /************** Integrate all spectra **************/

    /* Reflectance */
    let mut iso_integrated = iso_refl
        .iter()
        .map(|&s| camera.integrate(s * illuminant) * iso_weight);
    let mut my_integrated = my_reflectances
        .iter()
        .map(|&s| camera.integrate(s * illuminant) * my_spectra_weight);
    let mut locus_integrated = locus_spectra
        .iter()
        .map(|&s| camera.integrate(s) * locus_weight);

    iso_integrated
        .chain(my_integrated)
        .chain(locus_integrated)
        .chain(gradient_rb.into_iter())
        .collect()
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    // First argument after program name is the observer camera path
    let camera_response_path = if args.len() > 1 {
        &args[1]
    } else {
        eprintln!("Usage: {} <camera_response_path>", args[0]);
        std::process::exit(1);
    };

    // This is the 2006 2 degree CMF, also known by other names such as CIE 2015
    let target_cmf = load_response("colour/spectral_data/cmf/lin2012xyz2e_1_7sf.csv", ",");
    let camera_response = load_response(camera_response_path, " ");

    /* Illuminant */
    let illuminant_path = "colour/spectral_data/illuminants/CIE_std_illum_D50.csv";
    // let illuminant_path = "colour/spectral_data/illuminants/CIE_std_illum_A_1nm.csv";
    let illuminant = load_spectrum_iso(illuminant_path, ",").normalise().unwrap();

    let mut cam_rgb = generate_colours(camera_response.clone(), illuminant);
    let mut cmf_lms = generate_colours(target_cmf.clone(), illuminant);
    println!("Cam len new = {}", cam_rgb.len());

    let cam_rgb_avg = cam_rgb.iter().map(|x| x.magnitude()).sum::<f64>() / cam_rgb.len() as f64;
    let cmf_lms_avg = cmf_lms.iter().map(|x| x.magnitude()).sum::<f64>() / cmf_lms.len() as f64;
    for (cam, lms) in cam_rgb.iter_mut().zip(cmf_lms.iter_mut()) {
        *cam = *cam / cam_rgb_avg;
        *lms = *lms / cmf_lms_avg;
    }

    // pairs of in/outx
    let inoutpairs: Vec<InOutPair<_>> = cam_rgb
        .iter()
        .zip(cmf_lms.iter())
        .map(|(&input, &target)| InOutPair { input, target })
        .collect();

    /* Matrix first */
    let transform = TransformOptimiser {
        transform: MatrixTransform::<f64> {
            matrix: Matrix3x3::id(),
        },
    };
    let transform_parametric = transform.to_pars_unlocked();
    let refined = transform_parametric.refine(&inoutpairs, 1.0, 10).unwrap();
    println!("{:?}", refined.fmap(|x| x.value));
    println!(
        "Cam matrix: {:?}\n",
        refined.fmap(|x| x.value).transform.matrix.invert3x3()
    );

    let illuminant_cam_rgb = camera_response.integrate(illuminant).normalised();
    let gain = (illuminant_cam_rgb / illuminant_cam_rgb.magnitude()).map(|x| 1.0 / x);

    /* RPCC2 */
    let transform = TransformOptimiser {
        transform: Gained::<_, f64> {
            trans: Rpcc2::identity(),
            gain,
        },
    };
    let mut transform_parametric = transform.to_pars_unlocked();
    transform_parametric.transform.gain.lock();
    let refined2 = transform_parametric.refine(&inoutpairs, 1.0, 4).unwrap();
    println!("{:?}\n", refined2.fmap(|x| x.value));

    /* RPCC3 */
    let transform = TransformOptimiser {
        transform: refined2.transform.fmap(|x| x.value).convert::<Rpcc3<f64>>(),
    };
    let mut transform_parametric = transform.to_pars_unlocked();
    transform_parametric.transform.gain.lock();
    let refined3 = transform_parametric.refine(&inoutpairs, 1.0, 4).unwrap();
    println!("{:?}\n", refined3.fmap(|x| x.value));

    /* RPCC4 */
    let transform = TransformOptimiser {
        transform: refined3.transform.fmap(|x| x.value).convert::<Rpcc4<f64>>(),
    };
    let mut transform_parametric = transform.to_pars_unlocked();
    transform_parametric.transform.gain.lock();
    let refined4 = transform_parametric.refine(&inoutpairs, 1.0, 8).unwrap();
    println!("{:?}\n", refined4.fmap(|x| x.value));

    // Plot spectral locus in xy chromaticity
    use plotters::prelude::*;

    let matrix_transform = refined.fmap(|x| x.value).transform;
    let rpcc2_transform = refined2.fmap(|x| x.value).transform;
    let rpcc3_transform = refined3.fmap(|x| x.value).transform;
    let rpcc4_transform = refined4.fmap(|x| x.value).transform;
    let locus_wls: Vec<i32> = (430..=660).collect();

    // Monochromatic spikes for each wavelength
    let spikes: Vec<_> = locus_wls
        .iter()
        .map(|&wl| {
            SpectrumT::<f64>::from_fn(|w| {
                if (w - wl as f64).abs() < 1.5 {
                    1.0
                } else {
                    0.0
                }
            })
        })
        .collect();

    let to_xy = |xyz: Vector<f64, 3>| -> (f64, f64) {
        let s = xyz.0[0] + xyz.0[1] + xyz.0[2];
        if s > 1e-12 {
            (xyz.0[0] / s, xyz.0[1] / s)
        } else {
            (0.0, 0.0)
        }
    };

    // True XYZ locus
    let cmf_locus: Vec<_> = spikes
        .iter()
        .map(|s| to_xy(target_cmf.integrate(*s)))
        .collect();

    // Camera loci through each fitted transform
    let cam_rgb_spikes: Vec<_> = spikes
        .iter()
        .map(|s| camera_response.integrate(*s) / cam_rgb_avg)
        .collect();
    let locus_matrix: Vec<_> = cam_rgb_spikes
        .iter()
        .map(|&rgb| to_xy(matrix_transform.apply(rgb) * cmf_lms_avg))
        .collect();
    let locus_rpcc2: Vec<_> = cam_rgb_spikes
        .iter()
        .map(|&rgb| to_xy(rpcc2_transform.apply(rgb) * cmf_lms_avg))
        .collect();
    let locus_rpcc3: Vec<_> = cam_rgb_spikes
        .iter()
        .map(|&rgb| to_xy(rpcc3_transform.apply(rgb) * cmf_lms_avg))
        .collect();
    let locus_rpcc4: Vec<_> = cam_rgb_spikes
        .iter()
        .map(|&rgb| to_xy(rpcc4_transform.apply(rgb) * cmf_lms_avg))
        .collect();

    let root = BitMapBackend::new("spectral_locus_xy.png", (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Spectral Locus (xy chromaticity)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-0.05f64..0.85, -0.05f64..0.95)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(cmf_locus, &BLUE))
        .unwrap()
        .label("CIE XYZ CMF")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart
        .draw_series(LineSeries::new(locus_matrix, &BLACK))
        .unwrap()
        .label("Matrix")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));
    chart
        .draw_series(LineSeries::new(locus_rpcc2, &RED))
        .unwrap()
        .label("RPCC2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    chart
        .draw_series(LineSeries::new(locus_rpcc3, &GREEN))
        .unwrap()
        .label("RPCC3")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));
    chart
        .draw_series(LineSeries::new(locus_rpcc4, &MAGENTA))
        .unwrap()
        .label("RPCC4")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MAGENTA));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .unwrap();
    root.present().unwrap();
    println!("Saved plot to spectral_locus_xy.png");
}
