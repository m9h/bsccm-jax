// ==========================================================================
//  Shared LED-panel frame + parameters for the illumination heads.
//  No top-level geometry — `include <led_frame.scad>` from a head file.
//  Consumed by led_head.scad (generic post stand) and led_head_ofm.scad
//  (clips onto the OpenFlexure condenser dovetail).
// ==========================================================================

/* [Optics-driven] */
panel_height = 77.5;   // LED plane above the SAMPLE (led_array.recommend_height_mm)
mount_z      = 20;     // top of the generic base ring above the sample plane

/* [LED panel — measure your PCB] */
led_n     = 32;        // LEDs per side
led_pitch = 2.5;       // mm, LED centre spacing
pcb_t     = 1.6;       // PCB thickness
pcb_clear = 0.4;       // fit clearance around the PCB

/* [Head structure] */
wall         = 3;      // frame wall thickness
border       = 3;      // ledge width supporting the PCB (must be > 1.5)
diffuser_gap = 2;      // slot depth for a diffuser sheet under the LEDs
diffuser_t   = 1.0;    // diffuser sheet thickness
post_d       = 6;      // riser post diameter
base_wall    = 8;      // radial width of the generic base ring
screw_d      = 3.2;    // M3 clearance
$fn = 72;

/* [derived] */
led_active = led_n * led_pitch;                 // lit area (= light opening)
pcb_w      = led_active + 2*border;             // PCB footprint
frame_out  = pcb_w + 2*wall;                    // outer frame size
r_post     = (led_active/2 + frame_out/2) / 2;  // under the solid frame edge band
base_ir    = r_post - post_d/2 - 2;
base_or    = r_post + post_d/2 + base_wall;
frame_bottom = panel_height - diffuser_gap - diffuser_t;

module ring(or, ir, h) {
  difference() {
    cylinder(h = h, r = or);
    translate([0, 0, -0.1]) cylinder(h = h + 0.2, r = ir);
  }
}

// Frame that holds the panel LED-face-down. The central light opening
// (led_active) is smaller than the PCB, so the PCB rests on the border ledge
// and cannot fall through; the pocket captures it laterally.
module panel_frame() {
  frame_h = diffuser_gap + diffuser_t + pcb_t + wall;
  translate([-frame_out/2, -frame_out/2, frame_bottom])
  difference() {
    cube([frame_out, frame_out, frame_h]);
    // through light opening (LEDs shine down through this)
    translate([frame_out/2 - led_active/2, frame_out/2 - led_active/2, -0.1])
      cube([led_active, led_active, frame_h + 0.2]);
    // PCB pocket, open at the TOP, resting on the border ledge
    translate([wall - pcb_clear, wall - pcb_clear, diffuser_gap + diffuser_t])
      cube([pcb_w + 2*pcb_clear, pcb_w + 2*pcb_clear, pcb_t + wall + 0.2]);
    // diffuser slide-in slot, just under the LEDs
    translate([wall - 1, -0.1, diffuser_gap])
      cube([pcb_w + 2, wall + 1, diffuser_t]);
    // cable exit notch
    translate([frame_out/2 - 7, -0.1, diffuser_gap + diffuser_t])
      cube([14, wall + 0.2, pcb_t + wall]);
  }
}
