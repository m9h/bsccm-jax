// ==========================================================================
//  LED illumination head that CLIPS ONTO A STOCK OPENFLEXURE.
//
//  Reuses OpenFlexure's condenser dovetail (libs vendored in ofmlibs/, CERN-
//  OHL-W v2, (c) Richard Bowman) so the head slides onto the microscope's
//  illumination dovetail exactly like the stock condenser.  The rail is a
//  SLIDING fit with adjustment slots, so set the height at assembly and feed
//  the AS-BUILT panel-to-sample distance to led_array.LEDArray(height_mm=...).
//
//  Coordinates match led_frame.scad: sample plane at z=0, optical axis at
//  x=y=0, panel facing down.  OpenFlexure default: sample_z=75, the illumination
//  dovetail rail spans ~z2..z48 above the sample at y=35.
//
//  DERIVED WORK NOTICE: this file incorporates OpenFlexure geometry and is
//  therefore covered by CERN-OHL-W v2, separate from the repo's software licence.
// ==========================================================================

include <led_frame.scad>
use <ofmlibs/locking_dovetail.scad>

// The stock condenser's dovetail parameters (from openflexure illumination.scad).
function condenser_dovetail_params() = dovetail_params(
    overall_width  = 30,
    overall_height = 16,
    block_depth    = 16,
    taper_block    = true,
    nut_slot_slope = "down");

dt_y    = 35;   // OpenFlexure illumination dovetail mating-surface y
clamp_z = 15;   // where the clamp sits on the rail (rail spans ~z2..z48 here)

module ofm_mount() {
  // exact OpenFlexure condenser dovetail clamp, at the illumination dovetail
  translate([0, dt_y, clamp_z]) dovetail_clamp_m(condenser_dovetail_params());
  // rigid flaring riser: hull from the clamp top up to the panel's back edge
  hull() {
    translate([-15, 25, clamp_z + 14]) cube([30, 12, 2]);            // clamp-top slab
    translate([-frame_out/2, frame_out/2 - 8, frame_bottom]) cube([frame_out, 8, 3]);
  }
}

union() { panel_frame(); ofm_mount(); }
