use argmin::core::KV;
use argmin::core::observers::Observe;
use argmin::core::observers::ObserverMode;
use argmin::core::{CostFunction, Error, Executor, IterState, State};
use argmin::solver::neldermead::NelderMead;
use cmaes::{CMAESOptions, DVector};
use nalgebra::SVector;
use std::panic::catch_unwind;
use std::sync::mpsc::{Receiver, Sender};
use tap::Pipe;

use crate::sim::{Simulation, Trebutchet};

pub struct PendulumFit {
    target_path: Vec<SVector<f64, 2>>, // whatever your path looks like
    dt: f64,
}

fn make_simplex(initial: &[f64], step: f64) -> Vec<Vec<f64>> {
    let n = initial.len();
    let mut simplex = vec![initial.to_vec()]; // first vertex = initial guess

    for i in 0..n {
        let mut vertex = initial.to_vec();
        vertex[i] += step; // perturb one parameter
        simplex.push(vertex);
    }

    simplex // N+1 vertices total
}

struct ChannelObserver {
    cancel_rx: Receiver<()>,
    result_tx: Sender<Trebutchet>,
}

impl Observe<IterState<Vec<f64>, (), (), (), (), f64>> for ChannelObserver {
    fn observe_iter(
        &mut self,
        state: &IterState<Vec<f64>, (), (), (), (), f64>,
        _kv: &KV,
    ) -> Result<(), Error> {
        if self.cancel_rx.try_recv().is_ok() {
            return Err(Error::msg("cancelled"));
        }

        if state.is_best()
            && let Some(best) = &state.best_param
        {
            let _ = self.result_tx.send(Trebutchet::from_f64(best));
        }

        Ok(())
    }
}

impl PendulumFit {
    pub fn square(size: f64, time: f64, dt: f64, grace: f64) -> Self {
        let steps = (time / dt).round() as usize;

        // distribute points approximately evenly
        let side_steps = steps / 4;

        let mut target_path = Vec::with_capacity(side_steps * 4);

        for _ in (0..(grace / dt) as usize) {
            target_path.push(SVector::from([f64::INFINITY, f64::INFINITY]));
        }

        // right: (size,0) -> (size,size)
        for i in 0..side_steps {
            let t = i as f64 / side_steps as f64;
            target_path.push(SVector::from([size, size * t]));
        }

        // bottom: (0,0) -> (size,0)
        //for i in 0..side_steps {
        //    let t = i as f64 / side_steps as f64;
        //    target_path.push(SVector::from([size * t, 0.0]));
        //}

        // left: (0,size) -> (0,0)
        //for i in 0..side_steps {
        //    let t = i as f64 / side_steps as f64;
        //    target_path.push(SVector::from([0.0, size * (1.0 - t)]));
        //}

        Self { target_path, dt }
    }

    pub fn optimise_cmaes(self, treb: Trebutchet) -> (Sender<()>, Receiver<Trebutchet>) {
        let (cancel_tx, cancel_rx) = std::sync::mpsc::channel::<()>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<Trebutchet>();

        std::thread::spawn(move || {
            let sigma = 0.5;

            let initial = DVector::from_vec(treb.to_f64().to_vec());

            let mut cmaes = CMAESOptions::new(initial, sigma)
                .max_generations(10_000)
                .build(|x: &DVector<f64>| {
                    self.cost(&x.as_slice().to_vec()).unwrap_or(f64::INFINITY)
                })
                .unwrap();

            loop {
                if cancel_rx.try_recv().is_ok() {
                    break;
                }

                let terminated = cmaes.next().is_some();

                let best = &cmaes.overall_best_individual().unwrap();

                let treb = Trebutchet::from_f64(best.point.as_slice());

                let _ = result_tx.send(treb);

                if terminated {
                    break;
                }
            }
        });

        (cancel_tx, result_rx)
    }

    pub fn optimise(self, treb: Trebutchet) -> (Sender<()>, Receiver<Trebutchet>) {
        let (cancel_tx, cancel_rx) = std::sync::mpsc::channel::<()>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<Trebutchet>();

        let initial = treb.to_f64();

        std::thread::spawn(move || {
            let simplex = make_simplex(&initial, 0.1);

            let solver = NelderMead::new(simplex).with_sd_tolerance(1e-6).unwrap();

            let observer = ChannelObserver {
                cancel_rx,
                result_tx,
            };

            let _ = Executor::new(self, solver)
                .configure(|state| state.max_iters(10_000))
                .add_observer(observer, ObserverMode::Always)
                .run();
        });

        (cancel_tx, result_rx)
    }
}

impl CostFunction for PendulumFit {
    type Param = Vec<f64>; // your parameter vector
    type Output = f64; // scalar error

    fn cost(&self, params: &Vec<f64>) -> Result<f64, Error> {
        match catch_unwind(move || {
            let trebutchet = Trebutchet::from_f64(params);

            let (position, velocity) = trebutchet.initial();
            let mut sim = Simulation::<2, 4, _>::new(position, velocity, trebutchet);

            let mut error = 0.0;

            for target in self.target_path.iter() {
                if !sim.step_dt(self.dt) {
                    error += 1_000_000.0;
                }
                let (_, real) = trebutchet.carthesian(sim.q);
                if *target == SVector::from([f64::INFINITY, f64::INFINITY]) {
                    continue;
                }
                let diff = real - target;

                // position error
                let pos_err = diff.norm();
                error += pos_err;
            }

            Ok(error)
        }) {
            Ok(ok) => ok,
            Err(_) => Ok(100_000.0),
        }
    }
}
