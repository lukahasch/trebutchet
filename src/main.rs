use std::panic::catch_unwind;

use eframe::egui::{self, Color32, Slider};
use egui_plot::{Line, Plot, Points};
use trebutchet::{
    opt::PendulumFit,
    sim::{Simulation, Trebutchet},
};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        ..Default::default()
    };

    stacker::grow(2 * 1_000_000_000, || {
        let mut trebutchet = Trebutchet {
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
        let mut end_time = 2.0;
        let mut history = Vec::new();
        let mut cancel = None;
        let mut receive = None;

        eframe::run_ui_native("My egui App", options, move |ui, _frame| {
            let (_, arm2) = trebutchet.carthesian(sim.q);
            history.push([arm2[0], arm2[1]]);
            ui.request_repaint();
            egui::Panel::left("left").show_inside(ui, |ui| {
                ui.label("Simulation Settings");
                ui.add(
                    Slider::new(&mut sim.time_factor, 0.005..=10.0)
                        .logarithmic(true)
                        .text("Speed"),
                );
                ui.add(Slider::new(&mut end_time, 0.0..=5.0).text("End Time"));
                ui.label("Catapult Settings");
                ui.add(Slider::new(&mut trebutchet.arm_1_length, 0.0..=5.0).text("arm_1_length"));
                ui.add(Slider::new(&mut trebutchet.arm_2_length, 0.0..=5.0).text("arm_2_length"));
                ui.add(Slider::new(&mut trebutchet.arm_1_theta_0, 0.0..=5.0).text("arm_1_theta_0"));
                ui.add(Slider::new(&mut trebutchet.arm_2_theta_0, 0.0..=5.0).text("arm_2_theta_0"));
                ui.add(
                    Slider::new(&mut trebutchet.arm_2_theta_release, 0.0..=5.0)
                        .text("arm_2_theta_release"),
                );
                ui.add(Slider::new(&mut trebutchet.arm_1_mass, 0.0..=5.0).text("arm_1_mass"));
                ui.add(Slider::new(&mut trebutchet.arm_2_mass, 0.0..=5.0).text("arm_2_mass"));
                ui.add(
                    Slider::new(&mut trebutchet.projectile_mass, 0.0..=5.0).text("projectile_mass"),
                );
                if ui.button("Save").clicked() {
                    std::thread::spawn(move || match rfd::FileDialog::new().save_file() {
                        None => {}
                        Some(file) => {
                            std::fs::write(
                                file,
                                serde_json::to_string_pretty(&trebutchet).unwrap(),
                            )
                            .unwrap();
                        }
                    });
                }
                if ui.button("Open").clicked()
                    && let Some(file) = rfd::FileDialog::new().pick_file()
                {
                    match std::fs::read_to_string(file) {
                        Ok(s) => match serde_json::from_str(&s) {
                            Err(e) => eprintln!("{e:?}"),
                            Ok(treb) => {
                                trebutchet = treb;
                            }
                        },
                        Err(e) => eprintln!("{e:?}"),
                    }
                }
                if ui.button("opt::square").clicked() {
                    let (tx, rv) = PendulumFit::square(10.0, 1.6, 0.001, 0.2).optimise(trebutchet);
                    receive = Some(rv);
                    cancel = Some(tx);
                }
                if ui.button("opt::square (cmaes)").clicked() {
                    let (tx, rv) =
                        PendulumFit::square(10.0, 1.6, 0.001, 0.2).optimise_cmaes(trebutchet);
                    receive = Some(rv);
                    cancel = Some(tx);
                }
                if ui.button("opt::cancel").clicked()
                    && let Some(tx) = &cancel
                {
                    _ = tx.send(());
                    cancel = None;
                }
                if let Some(recv) = &receive {
                    while let Ok(treb) = recv.try_recv() {
                        let mut c = sim.clone();
                        c.l = treb;
                        if c.panics(end_time) {
                            continue;
                        }
                        let mut c = sim.clone();
                        c.l = treb;
                        let (position, velocity) = trebutchet.initial();
                        c.reset_with(position, velocity, 0.0);
                        if c.panics(end_time) {
                            continue;
                        }
                        trebutchet = treb;
                    }
                }
                sim.l = trebutchet;
                sim.step();
                if sim.time > end_time {
                    let (position, velocity) = trebutchet.initial();
                    sim.reset_with(position, velocity, 0.0);
                    history.clear();
                }
            });
            egui::CentralPanel::default().show_inside(ui, |ui| {
                let plot = Plot::new("animation").data_aspect(1.0);
                let (a, b) = sim.l.carthesian(sim.q);
                let points = Points::new("points", vec![[0.0, 0.0], [a[0], a[1]], [b[0], b[1]]])
                    .radius(20.0);
                let line = Line::new("line", vec![[0.0, 0.0], [a[0], a[1]], [b[0], b[1]]]);
                let history = Line::new("history", history.clone()).color(Color32::GOLD);
                plot.show(ui, |plot| {
                    plot.line(history);
                    plot.line(line);
                    plot.points(points);
                });
            });
        })
    })
}
