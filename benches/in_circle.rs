use criterion::{black_box, criterion_group, criterion_main, Criterion};
use float_expansion::in_circle;

use nalgebra::{Matrix2, Vector2};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand_distr::UnitCircle;
use rand_pcg::Pcg64;

type Vec2 = Vector2<f64>;
type Mtx2 = Matrix2<f64>;

const PCG_STATE: u128 = 0xcafef00dd15ea5e5;
const PCG_STREAM: u128 = 0xa02bdbf7bb3c0a7ac28fa16a64abf96;

fn grid(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10, 10);
    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(8).collect::<Vec<_>>();
            black_box([
                Vec2::new(vals[0] as f64, vals[1] as f64),
                Vec2::new(vals[2] as f64, vals[3] as f64),
                Vec2::new(vals[4] as f64, vals[5] as f64),
                Vec2::new(vals[6] as f64, vals[7] as f64),
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("in_circle_grid", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d]| {
                in_circle(*a, *b, *c, *d);
            })
        })
    });
}

fn uniform_random(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10.0, 10.0);
    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(8).collect::<Vec<_>>();
            black_box([
                Vec2::new(vals[0], vals[1]),
                Vec2::new(vals[2], vals[3]),
                Vec2::new(vals[4], vals[5]),
                Vec2::new(vals[6], vals[7]),
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("in_circle_uniform_random", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d]| {
                in_circle(*a, *b, *c, *d);
            })
        })
    });
}

fn near_cocircular(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10.0, 10.0);

    let data = (0..10000)
        .map(|_| {
            let vals = UnitCircle.sample_iter(&mut rng).take(4).collect::<Vec<_>>();
            let radius = dist.sample(&mut rng);
            let x = dist.sample(&mut rng);
            let y = dist.sample(&mut rng);
            let offset = Vec2::new(x, y);

            black_box([
                Vec2::new(vals[0][0], vals[0][1]) * radius + offset,
                Vec2::new(vals[1][0], vals[1][1]) * radius + offset,
                Vec2::new(vals[2][0], vals[2][1]) * radius + offset,
                Vec2::new(vals[3][0], vals[3][1]) * radius + offset,
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("in_circle_near_cocircular", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d]| {
                in_circle(*a, *b, *c, *d);
            })
        })
    });
}

fn cocircular(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-40960, 40960);
    let abs = Uniform::new_inclusive(0, 40960);
    let mtxs = [
        Mtx2::new(1.0, 0.0, 0.0, 1.0),
        Mtx2::new(-1.0, 0.0, 0.0, 1.0),
        Mtx2::new(0.0, -1.0, 1.0, 0.0),
        Mtx2::new(0.0, 1.0, 1.0, 0.0),
        Mtx2::new(-1.0, 0.0, 0.0, -1.0),
        Mtx2::new(1.0, 0.0, 0.0, -1.0),
        Mtx2::new(0.0, 1.0, -1.0, 0.0),
        Mtx2::new(0.0, -1.0, -1.0, 0.0),
    ];

    let data = (0..10000)
        .map(|_| {
            let x = abs.sample(&mut rng) as f64 / 4096.0;
            let y = abs.sample(&mut rng) as f64 / 4096.0;
            let v = Vec2::new(x, y);
            let cx = dist.sample(&mut rng) as f64 / 4096.0;
            let cy = dist.sample(&mut rng) as f64 / 4096.0;
            let offset = Vec2::new(cx, cy);

            let mut vals = mtxs.choose_multiple(&mut rng, 4);
            black_box([
                vals.next().unwrap() * v + offset,
                vals.next().unwrap() * v + offset,
                vals.next().unwrap() * v + offset,
                vals.next().unwrap() * v + offset,
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("in_circle_cocircular", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d]| {
                in_circle(*a, *b, *c, *d);
            })
        })
    });
}

criterion_group!(
    benches_in_circle,
    grid,
    uniform_random,
    near_cocircular,
    cocircular
);
criterion_main!(benches_in_circle);
