// ==========================================================================
//  LED illumination head — GENERIC post stand (no OpenFlexure dependency).
//  Panel faces DOWN, LED plane `panel_height` mm above the sample, coaxial
//  with the optical axis.  Mounts via three M3 holes in the base ring.
//
//  For a head that clips onto a stock OpenFlexure, use led_head_ofm.scad.
//  `panel_height` comes from bsccm_jax.led_array.recommend_height_mm — run
//  hardware/gen_led_head.py to fill it for your objective.
// ==========================================================================

include <led_frame.scad>

module posts() {
  z0 = mount_z - base_wall/2;                 // merge into the base ring
  z1 = frame_bottom + 2;                       // overlap into the frame
  // edge-midpoint angles sit under the solid frame band (not the aperture)
  for (a = [0, 90, 180, 270])
    translate([r_post*cos(a), r_post*sin(a), z0])
      cylinder(h = z1 - z0, r = post_d/2);
}

module base_ring() {
  difference() {
    translate([0, 0, mount_z - base_wall/2]) ring(base_or, base_ir, base_wall);
    for (a = [45, 165, 285])
      translate([(base_ir + base_or)/2 * cos(a),
                 (base_ir + base_or)/2 * sin(a), mount_z - base_wall])
        cylinder(h = base_wall * 2, r = screw_d/2);
  }
}

module led_head() { union() { panel_frame(); posts(); base_ring(); } }

led_head();
