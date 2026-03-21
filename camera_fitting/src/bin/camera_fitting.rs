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

fn load_all_iso_spectra(base_path: &str) -> Vec<SpectrumT<f64>> {
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

pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    // First argument after program name is the observer camera path
    let observer_camera_path = if args.len() > 1 {
        &args[1]
    } else {
        eprintln!("Usage: {} <camera_response_path>", args[0]);
        std::process::exit(1);
    };

    // This is the 2006 2 degree CMF, also known by other names such as CIE 2015
    let cmf = load_response("colour/spectral_data/cmf/lin2012xyz2e_1_7sf.csv", ",");

    let observer_camera = load_response(observer_camera_path, " ");

    // Load spectral reflectances
    let mut iso_reflectances = load_all_iso_spectra(&format!(
        "{SPECTRAL_MEASUREMENTS_PATH}/Data/Spectra/ReflectanceISO"
    ));

    let mut all_reflectances = vec![];
    all_reflectances.append(&mut iso_reflectances.clone());

    /* Illuminant */
    let illuminant_path = "colour/spectral_data/illuminants/CIE_std_illum_D50.csv";
    // let illuminant_path = "colour/spectral_data/illuminants/CIE_std_illum_A_1nm.csv";

    let illuminant = load_spectrum_iso(illuminant_path, ",").normalise().unwrap();

    // apply illuminant to reflactrances
    let mut all_spectra = all_reflectances
        .into_iter()
        .map(|refl| refl * illuminant)
        .collect::<Vec<_>>();

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

    all_spectra.append(&mut sharp_spectral);
    all_spectra.append(&mut medium_spectral);
    all_spectra.append(&mut soft_spectral);

    /* Generate a red-blue gradient for smooth luminance in that region */
    let redblue_steps = 15;
    let rb_to_white_steps = 8;
    let max_rb_white = 0.4;
    let reblue_weight = 15.0;
    let bluelight: SpectrumT<f64> = GaussianSpectrum {
        cwl: 470.,
        fwhm: 12.0,
    }
    .discretise()
    .normalise_area(reblue_weight);
    let redlight: SpectrumT<f64> = GaussianSpectrum {
        cwl: 630.,
        fwhm: 12.0,
    }
    .discretise()
    .normalise_area(reblue_weight);
    let whitelight = illuminant.normalise_area(1.0);
    let mut redblue = (0..redblue_steps)
        .map(|step| {
            let a = (step as f64 / (redblue_steps - 1) as f64);
            return bluelight * a + redlight * (1.0 - a);
        })
        .collect::<Vec<_>>();
    // also blend to white
    let with_white = (0..rb_to_white_steps)
        .map(|step| {
            let a = (step as f64 / (redblue_steps - 1) as f64) * max_rb_white;
            redblue
                .iter()
                .map(|rbspectrum| {
                    return (whitelight * a + *rbspectrum * (1.0 - a)) / rb_to_white_steps as f64;
                })
                .collect::<Vec<_>>()
                .into_iter()
        })
        .flatten()
        .collect::<Vec<_>>();

    all_spectra.append(&mut redblue);

    println!(
        "{:?}",
        observer_camera.integrate(SpectrumT::<f64>::from_fn(|_wl| 1.0))
    );
    println!("{:?}", cmf.integrate(SpectrumT::<f64>::from_fn(|_wl| 1.0)));

    println!("Camera:");
    observer_camera.print_peak_values();

    println!("Human LMS");
    cmf.print_peak_values();

    let mut cam_rgb = all_spectra
        .iter()
        .map(|&spectrum| observer_camera.integrate(spectrum))
        .collect::<Vec<_>>();

    let mut cmf_lms = all_spectra
        .iter()
        .map(|&spectrum| cmf.integrate(spectrum))
        .collect::<Vec<_>>();

    let cam_rgb_avg = cam_rgb.iter().map(|x| x.magnitude()).sum::<f64>() / cam_rgb.len() as f64;
    let cmf_lms_avg = cmf_lms.iter().map(|x| x.magnitude()).sum::<f64>() / cmf_lms.len() as f64;
    for (cam, lms) in cam_rgb.iter_mut().zip(cmf_lms.iter_mut()) {
        *cam = *cam / cam_rgb_avg;
        *lms = *lms / cmf_lms_avg;
    }

    // pairs of in/out
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
    let refined = transform_parametric.refine(&inoutpairs, 0.3, 300).unwrap();
    println!("{:?}", refined.fmap(|x| x.value));
    println!(
        "Cam matrix: {:?}\n",
        refined.fmap(|x| x.value).transform.matrix.invert3x3()
    );

    let illuminant_cam_rgb = observer_camera.integrate(illuminant).normalised();
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
    let refined2 = transform_parametric.refine(&inoutpairs, 0.05, 300).unwrap();
    println!("{:?}\n", refined2.fmap(|x| x.value));

    /* RPCC3 */
    let transform = TransformOptimiser {
        transform: refined2.transform.fmap(|x| x.value).convert::<Rpcc3<f64>>(),
    };
    let mut transform_parametric = transform.to_pars_unlocked();
    transform_parametric.transform.gain.lock();
    let refined3 = transform_parametric
        .refine(&inoutpairs, 0.025, 100)
        .unwrap();
    println!("{:?}\n", refined3.fmap(|x| x.value));

    /* RPCC4 */
    let transform = TransformOptimiser {
        transform: refined3.transform.fmap(|x| x.value).convert::<Rpcc4<f64>>(),
    };
    let mut transform_parametric = transform.to_pars_unlocked();
    transform_parametric.transform.gain.lock();
    let refined4 = transform_parametric.refine(&inoutpairs, 0.02, 10).unwrap();
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
    let cmf_locus: Vec<_> = spikes.iter().map(|s| to_xy(cmf.integrate(*s))).collect();

    // Camera loci through each fitted transform
    let cam_rgb_spikes: Vec<_> = spikes
        .iter()
        .map(|s| observer_camera.integrate(*s) / cam_rgb_avg)
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
