use eframe::egui;
use egui_plot::{Line, Plot, Points};
use trebutchet::sim::{Simulation, Trebutchet};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        ..Default::default()
    };

    stacker::grow(2 * 1_000_000_000, || {
        let trebutchet = Trebutchet {
            arm_1_length: 0.80,
            arm_2_length: 1.60,
            arm_1_theta_0: 0.0,
            arm_2_theta_0: 0.0,
            arm_2_theta_release: 45.0,
            arm_1_mass: 1.0,
            arm_2_mass: 0.1,
            projectile_mass: 10.0,
        };
        let (position, velocity) = trebutchet.initial();
        let mut sim = Simulation::<2, 4, _>::new(position, velocity, trebutchet);

        eframe::run_ui_native("My egui App", options, move |ui, _frame| {
            ui.request_repaint();
            ui.add(
                egui::Slider::new(&mut sim.time_factor, 0.005..=10.0)
                    .logarithmic(true)
                    .text("Speed"),
            );
            sim.step();
            egui::CentralPanel::default().show_inside(ui, |ui| {
                let plot = Plot::new("animation").data_aspect(1.0);
                let (a, b) = sim.l.carthesian(sim.q);
                let points = Points::new("points", vec![[0.0, 0.0], [a[0], a[1]], [b[0], b[1]]])
                    .radius(20.0);
                let line = Line::new("line", vec![[0.0, 0.0], [a[0], a[1]], [b[0], b[1]]]);
                plot.show(ui, |plot| {
                    plot.line(line);
                    plot.points(points);
                });
            });
        })
    })
}
