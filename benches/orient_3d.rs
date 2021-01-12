use criterion::{black_box, criterion_group, criterion_main, Criterion};
use robust_geo::orient_3d;

use nalgebra::Vector3;
use rand::distributions::{Distribution, Uniform};
use rand_pcg::Pcg64;

type Vec3 = Vector3<f64>;

const PCG_STATE: u128 = 0xcafef00dd15ea5e5;
const PCG_STREAM: u128 = 0xa02bdbf7bb3c0a7ac28fa16a64abf96;

fn grid(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10, 10);
    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(12).collect::<Vec<_>>();
            black_box([
                Vec3::new(vals[0] as f64, vals[1] as f64, vals[2] as f64),
                Vec3::new(vals[3] as f64, vals[4] as f64, vals[5] as f64),
                Vec3::new(vals[6] as f64, vals[7] as f64, vals[8] as f64),
                Vec3::new(vals[9] as f64, vals[10] as f64, vals[11] as f64),
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("orient_3d_grid", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d]| {
                orient_3d(*a, *b, *c, *d);
            })
        })
    });
}

fn uniform_random(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10.0, 10.0);
    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(12).collect::<Vec<_>>();
            black_box([
                Vec3::new(vals[0], vals[1], vals[2]),
                Vec3::new(vals[3], vals[4], vals[5]),
                Vec3::new(vals[6], vals[7], vals[8]),
                Vec3::new(vals[9], vals[10], vals[11]),
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("orient_3d_uniform_random", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d]| {
                orient_3d(*a, *b, *c, *d);
            })
        })
    });
}

fn near_coplanar(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10.0, 10.0);
    let fac_dist = Uniform::new(-1.0, 2.0);

    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(9).collect::<Vec<_>>();
            let a = Vec3::new(vals[0], vals[1], vals[2]);
            let b = Vec3::new(vals[3], vals[4], vals[5]);
            let c = Vec3::new(vals[6], vals[7], vals[8]);

            let fac = fac_dist.sample(&mut rng);
            let fac2 = fac_dist.sample(&mut rng);
            black_box([a, b, c, c + (a + (b - a) * fac - c) * fac2])
        })
        .collect::<Vec<_>>();

    c.bench_function("orient_3d_near_coplanar", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d]| {
                orient_3d(*a, *b, *c, *d);
            })
        })
    });
}

fn coplanar(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-40960, 40960);
    let fac_dist = Uniform::new(-4095, 4096);

    let data = (0..10000)
        .map(|_| {
            let vals = dist
                .sample_iter(&mut rng)
                .take(9)
                .map(|x| (x as f64) / 4096.0)
                .collect::<Vec<_>>();
            let a = Vec3::new(vals[0], vals[1], vals[2]);
            let b = Vec3::new(vals[3], vals[4], vals[5]);
            let c = Vec3::new(vals[6], vals[7], vals[8]);

            let fac = fac_dist.sample(&mut rng) as f64 / 16.0;
            let fac2 = fac_dist.sample(&mut rng) as f64 / 16.0;
            black_box([a, b, c, c + (a + (b - a) * fac - c) * fac2])
        })
        .collect::<Vec<_>>();

    c.bench_function("orient_3d_coplanar", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d]| {
                orient_3d(*a, *b, *c, *d);
            })
        })
    });
}

criterion_group!(
    benches_orient_3d,
    grid,
    uniform_random,
    near_coplanar,
    coplanar
);
criterion_main!(benches_orient_3d);
