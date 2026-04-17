use nalgebra::{Const, DimMin, DimSub, SVector};
use num_dual::{DualNum, hessian};
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub trait Lagrangian<const D: usize> {
    fn lagrangian<T>(&self, q: SVector<T, D>, v: SVector<T, D>) -> T
    where
        T: DualNum<f64> + Copy;
    fn ext_force<T>(&self, q: SVector<T, D>, v: SVector<T, D>) -> SVector<T, D>
    where
        T: DualNum<f64> + Copy;
}

pub struct Simulation<const D: usize, const T: usize, L>
where
    L: Lagrangian<D>,
{
    pub q: SVector<f64, D>,
    pub v: SVector<f64, D>,
    pub l: L,
    pub time_factor: f64,
    last: Instant,
}

impl<const D: usize, const T: usize, L> Simulation<D, T, L>
where
    L: Lagrangian<D> + 'static,
    Const<D>: DimMin<Const<D>, Output = Const<D>> + DimSub<Const<1>>,
{
    pub fn new(q: SVector<f64, D>, v: SVector<f64, D>, l: L) -> Self {
        Self {
            q,
            v,
            l,
            time_factor: 1.0,
            last: Instant::now(),
        }
    }

    pub fn step(&mut self) -> &mut Self {
        let dt = self.last.elapsed().as_secs_f64() * self.time_factor;
        self.step_dt(dt);
        self.last = Instant::now();
        self
    }

    pub fn step_dt(&mut self, mut dt: f64) {
        while dt > 0.0 {
            let (q, v) = step_lagrangian::<D, T, _>(dt.min(0.005), self.q, self.v, &self.l);

            if q.as_slice().iter().any(|x| x.is_nan()) || v.as_slice().iter().any(|x| x.is_nan()) {
                println!(
                    "NAN detected at dt: {dt} | q: {:?}, v: {:?}",
                    self.q, self.v
                );
                break;
            } else {
                self.q = q;
                self.v = v;
            }
            dt -= 0.005;
        }
    }
}

/// Computes the next state using a generalized Lagrangian solver via RK4 integration.
pub fn step_lagrangian<const D: usize, const T: usize, L>(
    dt: f64,
    q: SVector<f64, D>,
    v: SVector<f64, D>,
    lagrangian: &L,
) -> (SVector<f64, D>, SVector<f64, D>)
where
    L: Lagrangian<D> + 'static,
    Const<D>: DimMin<Const<D>, Output = Const<D>> + DimSub<Const<1>>,
{
    // Helper closure to compute acceleration (q_ddot) for any given state (q, v)
    let calc_accel = |q_curr: SVector<f64, D>, v_curr: SVector<f64, D>| -> SVector<f64, D> {
        let mut state_combined = SVector::<f64, T>::zeros();
        state_combined
            .fixed_view_mut::<D, 1>(0, 0)
            .copy_from(&q_curr);
        state_combined
            .fixed_view_mut::<D, 1>(D, 0)
            .copy_from(&v_curr);

        // Compute the Hessian of the Lagrangian
        let (_, grad, hess) = hessian(
            |x| {
                let q_dual = x.fixed_view::<D, 1>(0, 0).into_owned();
                let v_dual = x.fixed_view::<D, 1>(D, 0).into_owned();
                lagrangian.lagrangian(q_dual, v_dual)
            },
            &state_combined,
        );

        let dl_dq = grad.fixed_view::<D, 1>(0, 0);
        let mass_matrix = hess.fixed_view::<D, D>(D, D);
        let d2l_dvdq = hess.fixed_view::<D, D>(D, 0);

        // Solve the Euler-Lagrange equation for acceleration (q_ddot)
        let external = lagrangian.ext_force(q_curr, v_curr);
        let rhs = dl_dq - (d2l_dvdq * v_curr) + external;

        mass_matrix
            .full_piv_lu()
            .solve(&rhs)
            .expect("Singular Mass Matrix: Check Lagrangian definition or constraints")
    };

    // --- Runge-Kutta 4 (RK4) Integration ---

    // k1: Evaluate at the start of the interval
    let k1_v = calc_accel(q, v);
    let k1_q = v;

    // k2: Evaluate at the midpoint using k1
    let k2_v = calc_accel(q + k1_q * (dt / 2.0), v + k1_v * (dt / 2.0));
    let k2_q = v + k1_v * (dt / 2.0);

    // k3: Evaluate at the midpoint using k2
    let k3_v = calc_accel(q + k2_q * (dt / 2.0), v + k2_v * (dt / 2.0));
    let k3_q = v + k2_v * (dt / 2.0);

    // k4: Evaluate at the end using k3
    let k4_v = calc_accel(q + k3_q * dt, v + k3_v * dt);
    let k4_q = v + k3_v * dt;

    // Combine to find the next state
    let next_q = q + (k1_q + k2_q * 2.0 + k3_q * 2.0 + k4_q) * (dt / 6.0);
    let next_v = v + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * (dt / 6.0);

    (next_q, next_v)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Trebutchet {
    pub arm_1_length: f64,
    pub arm_2_length: f64,
    pub arm_1_theta_0: f64,
    pub arm_2_theta_0: f64,
    pub arm_2_theta_release: f64,
    pub arm_1_mass: f64,
    pub arm_2_mass: f64,
    pub projectile_mass: f64,
}

impl Trebutchet {
    pub fn to_f64(&self) -> SVector<f64, 8> {
        SVector::from([
            self.arm_1_length,
            self.arm_2_length,
            self.arm_1_theta_0,
            self.arm_2_theta_0,
            self.arm_2_theta_release,
            self.arm_1_mass,
            self.arm_2_mass,
            self.projectile_mass,
        ])
    }

    pub fn from_f64(state: SVector<f64, 8>) -> Self {
        Self {
            arm_1_length: state[0],
            arm_2_length: state[1],
            arm_1_theta_0: state[2],
            arm_2_theta_0: state[3],
            arm_2_theta_release: state[4],
            arm_1_mass: state[5],
            arm_2_mass: state[6],
            projectile_mass: state[7],
        }
    }

    pub fn initial(&self) -> (SVector<f64, 2>, SVector<f64, 2>) {
        (
            SVector::from([self.arm_1_theta_0, self.arm_2_theta_0]),
            SVector::zeros(),
        )
    }

    pub fn carthesian_arm_1<T>(&self, q: SVector<T, 2>, len: f64) -> SVector<T, 2>
    where
        T: DualNum<f64> + Copy,
    {
        // FIX: Enforce meters for internal physics calculations
        let l = T::from(len);
        let x_1 = q[0].cos() * l;
        let y_1 = q[0].sin() * l;
        SVector::from([x_1, y_1])
    }

    pub fn carthesian_arm_2<T>(&self, q: SVector<T, 2>, len: f64) -> SVector<T, 2>
    where
        T: DualNum<f64> + Copy,
    {
        // FIX: Enforce meters for internal physics calculations
        let l1 = T::from(self.arm_1_length);
        let l2 = T::from(len);

        let x_1 = q[0].cos() * l1;
        let y_1 = q[0].sin() * l1;
        let x_2 = (q[0] + q[1]).cos() * l2 + x_1;
        let y_2 = (q[0] + q[1]).sin() * l2 + y_1;
        SVector::from([x_2, y_2])
    }

    pub fn carthesian(&self, q: SVector<f64, 2>) -> (SVector<f64, 2>, SVector<f64, 2>) {
        (
            self.carthesian_arm_1(q, self.arm_1_length),
            self.carthesian_arm_2(q, self.arm_2_length),
        )
    }

    pub fn kinetic<T>(&self, q: SVector<T, 2>, v: SVector<T, 2>) -> T
    where
        T: DualNum<f64> + Copy,
    {
        let m1 = T::from(self.arm_1_mass);
        let m2 = T::from(self.arm_2_mass);
        let mp = T::from(self.projectile_mass);

        let l1 = T::from(self.arm_1_length);
        let l2 = T::from(self.arm_2_length);

        let q1 = q[1];
        let v0 = v[0];
        let v1 = v[1];

        let arm_1_k = T::from(1.0 / 6.0) * m1 * l1.powi(2) * v0.powi(2);

        let v_cm2_sq = l1.powi(2) * v0.powi(2)
            + (l2 / 2.0).powi(2) * (v0 + v1).powi(2)
            + l1 * l2 * v0 * (v0 + v1) * q1.cos();

        let arm_2_trans_k = T::from(0.5) * m2 * v_cm2_sq;

        let arm_2_rot_k = T::from(1.0 / 24.0) * m2 * l2.powi(2) * (v0 + v1).powi(2);

        let arm_2_k = arm_2_trans_k + arm_2_rot_k;

        let v_p_sq = l1.powi(2) * v0.powi(2)
            + l2.powi(2) * (v0 + v1).powi(2)
            + T::from(2.0) * l1 * l2 * v0 * (v0 + v1) * q1.cos();

        let projectile_k = T::from(0.5) * mp * v_p_sq;

        arm_1_k + arm_2_k + projectile_k
    }

    pub fn gravity() -> f64 {
        9.81
    }

    pub fn potential<T>(&self, q: SVector<T, 2>) -> T
    where
        T: DualNum<f64> + Copy,
    {
        let g = T::from(Self::gravity());

        let m1 = T::from(self.arm_1_mass);
        let m2 = T::from(self.arm_2_mass);
        let mp = T::from(self.projectile_mass);

        let y_cm1 = self.carthesian_arm_1(q, self.arm_1_length / 2.0)[1];
        let y_cm2 = self.carthesian_arm_2(q, self.arm_2_length / 2.0)[1];
        let y_p = self.carthesian_arm_2(q, self.arm_2_length)[1];

        (m1 * y_cm1 + m2 * y_cm2 + mp * y_p) * g
    }
}

impl Lagrangian<2> for Trebutchet {
    fn lagrangian<T>(&self, q: SVector<T, 2>, v: SVector<T, 2>) -> T
    where
        T: DualNum<f64> + Copy,
    {
        let kinetic = self.kinetic(q, v);
        let potential = self.potential(q);
        kinetic - potential
    }

    fn ext_force<T>(&self, _q: SVector<T, 2>, _v: SVector<T, 2>) -> SVector<T, 2>
    where
        T: DualNum<f64> + Copy,
    {
        SVector::zeros()
    }
}
