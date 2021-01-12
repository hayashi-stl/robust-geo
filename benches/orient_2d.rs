use criterion::{black_box, criterion_group, criterion_main, Criterion};
use robust_geo::orient_2d;

use nalgebra::Vector2;
use rand::distributions::{Distribution, Uniform};
use rand_pcg::Pcg64;

type Vec2 = Vector2<f64>;

const PCG_STATE: u128 = 0xcafef00dd15ea5e5;
const PCG_STREAM: u128 = 0xa02bdbf7bb3c0a7ac28fa16a64abf96;

fn grid(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10, 10);
    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(6).collect::<Vec<_>>();
            black_box([
                Vec2::new(vals[0] as f64, vals[1] as f64),
                Vec2::new(vals[2] as f64, vals[3] as f64),
                Vec2::new(vals[4] as f64, vals[5] as f64),
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("orient_2d_grid", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c]| {
                orient_2d(*a, *b, *c);
            })
        })
    });
}

fn uniform_random(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10.0, 10.0);
    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(6).collect::<Vec<_>>();
            black_box([
                Vec2::new(vals[0], vals[1]),
                Vec2::new(vals[2], vals[3]),
                Vec2::new(vals[4], vals[5]),
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("orient_2d_uniform_random", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c]| {
                orient_2d(*a, *b, *c);
            })
        })
    });
}

fn near_collinear(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10.0, 10.0);
    let fac_dist = Uniform::new(-1.0, 2.0);

    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(6).collect::<Vec<_>>();
            let fac = fac_dist.sample(&mut rng);
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);
            black_box([a, b, a + (b - a) * fac])
        })
        .collect::<Vec<_>>();

    c.bench_function("orient_2d_near_collinear", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c]| {
                orient_2d(*a, *b, *c);
            })
        })
    });
}

fn collinear(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-40960, 40960);
    let fac_dist = Uniform::new(-4095, 4096);

    let data = (0..10000)
        .map(|_| {
            let vals = dist
                .sample_iter(&mut rng)
                .take(4)
                .map(|x| (x as f64) / 4096.0)
                .collect::<Vec<_>>();
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);

            let fac = fac_dist.sample(&mut rng) as f64 / 16.0;
            black_box([a, b, a + (b - a) * fac])
        })
        .collect::<Vec<_>>();

    c.bench_function("orient_2d_collinear", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c]| {
                orient_2d(*a, *b, *c);
            })
        })
    });
}

criterion_group!(
    benches_orient_2d,
    grid,
    uniform_random,
    near_collinear,
    collinear
);
criterion_main!(benches_orient_2d);
