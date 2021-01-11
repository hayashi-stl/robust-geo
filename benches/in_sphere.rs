use criterion::{black_box, criterion_group, criterion_main, Criterion};
use float_expansion::in_sphere;

use nalgebra::{Matrix3, Vector3};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand_distr::UnitSphere;
use rand_pcg::Pcg64;

type Vec3 = Vector3<f64>;
type Mtx3 = Matrix3<f64>;

const PCG_STATE: u128 = 0xcafef00dd15ea5e5;
const PCG_STREAM: u128 = 0xa02bdbf7bb3c0a7ac28fa16a64abf96;

fn grid(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10, 10);
    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(15).collect::<Vec<_>>();
            black_box([
                Vec3::new(vals[0] as f64, vals[1] as f64, vals[2] as f64),
                Vec3::new(vals[3] as f64, vals[4] as f64, vals[5] as f64),
                Vec3::new(vals[6] as f64, vals[7] as f64, vals[8] as f64),
                Vec3::new(vals[9] as f64, vals[10] as f64, vals[11] as f64),
                Vec3::new(vals[12] as f64, vals[13] as f64, vals[14] as f64),
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("in_sphere_grid", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d, e]| {
                in_sphere(*a, *b, *c, *d, *e);
            })
        })
    });
}

fn uniform_random(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10.0, 10.0);
    let data = (0..10000)
        .map(|_| {
            let vals = dist.sample_iter(&mut rng).take(15).collect::<Vec<_>>();
            black_box([
                Vec3::new(vals[0], vals[1], vals[2]),
                Vec3::new(vals[3], vals[4], vals[5]),
                Vec3::new(vals[6], vals[7], vals[8]),
                Vec3::new(vals[9], vals[10], vals[11]),
                Vec3::new(vals[12], vals[13], vals[14]),
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("in_sphere_uniform_random", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d, e]| {
                in_sphere(*a, *b, *c, *d, *e);
            })
        })
    });
}

fn near_cospherical(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-10.0, 10.0);

    let data = (0..10000)
        .map(|_| {
            let vals = UnitSphere.sample_iter(&mut rng).take(5).collect::<Vec<_>>();
            let radius = dist.sample(&mut rng);
            let x = dist.sample(&mut rng);
            let y = dist.sample(&mut rng);
            let z = dist.sample(&mut rng);
            let offset = Vec3::new(x, y, z);

            black_box([
                Vec3::new(vals[0][0], vals[0][1], vals[0][2]) * radius + offset,
                Vec3::new(vals[1][0], vals[1][1], vals[1][2]) * radius + offset,
                Vec3::new(vals[2][0], vals[2][1], vals[2][2]) * radius + offset,
                Vec3::new(vals[3][0], vals[3][1], vals[3][2]) * radius + offset,
                Vec3::new(vals[4][0], vals[4][1], vals[4][2]) * radius + offset,
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("in_sphere_near_cospherical", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d, e]| {
                in_sphere(*a, *b, *c, *d, *e);
            })
        })
    });
}

fn cospherical(c: &mut Criterion) {
    let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
    let dist = Uniform::new_inclusive(-40960, 40960);
    let abs = Uniform::new_inclusive(0, 40960);
    let flip = Mtx3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let rot4 = Mtx3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let rot3 = Mtx3::new(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0);
    let mut mtxs = vec![];

    for i in 0..2 {
        for j in 0..4 {
            for k in 0..3 {
                mtxs.push(
                    std::iter::repeat(rot3).take(k).product::<Mtx3>()
                        * std::iter::repeat(rot4).take(j).product::<Mtx3>()
                        * std::iter::repeat(flip).take(i).product::<Mtx3>(),
                );
            }
        }
    }

    let data = (0..10000)
        .map(|_| {
            let x = abs.sample(&mut rng) as f64 / 4096.0;
            let y = abs.sample(&mut rng) as f64 / 4096.0;
            let z = abs.sample(&mut rng) as f64 / 4096.0;
            let v = Vec3::new(x, y, z);
            let cx = dist.sample(&mut rng) as f64 / 4096.0;
            let cy = dist.sample(&mut rng) as f64 / 4096.0;
            let cz = dist.sample(&mut rng) as f64 / 4096.0;
            let offset = Vec3::new(cx, cy, cz);

            let mut vals = mtxs.choose_multiple(&mut rng, 5);
            black_box([
                vals.next().unwrap() * v + offset,
                vals.next().unwrap() * v + offset,
                vals.next().unwrap() * v + offset,
                vals.next().unwrap() * v + offset,
                vals.next().unwrap() * v + offset,
            ])
        })
        .collect::<Vec<_>>();

    c.bench_function("in_sphere_cospherical", move |b| {
        b.iter(|| {
            data.iter().for_each(|[a, b, c, d, e]| {
                in_sphere(*a, *b, *c, *d, *e);
            })
        })
    });
}

criterion_group!(
    benches_in_sphere,
    grid,
    uniform_random,
    near_cospherical,
    cospherical
);
criterion_main!(benches_in_sphere);
