/* Graph Foundry · Cinema skill (cortex-cinema.js)
 * A shared cinematographer: slow auto-orbit, eased framing of the lead
 * individual, and a zoom-to-best push-in. Pure three.js, no dependencies. */
(function (root) {
  "use strict";
  function CortexCinema(camera) {
    this.cam = camera;
    this.center = new THREE.Vector3(0, 2, 0);
    this.radius = 30; this.targetRadius = 30;
    this.height = 16; this.targetHeight = 16;
    this.angle = 0; this.spin = 0.0022;
    this.look = new THREE.Vector3(0, 2, 0);
    this.targetLook = new THREE.Vector3(0, 2, 0);
    this.zoomed = false;
    // Cinematography: cycle through framed shots for motion variety.
    this.shots = [
      { radius: 30, height: 16, spin: 0.0022 },   // wide establishing orbit
      { radius: 22, height: 7, spin: 0.0014 },    // low dramatic sweep
      { radius: 26, height: 26, spin: 0.0026 },   // high overhead
      { radius: 18, height: 11, spin: 0.0018 }    // medium push
    ];
    this.shot = 0; this.shotT = 0; this.shotLen = 520;
  }
  CortexCinema.prototype.frameLead = function (pos) {
    if (pos) this.targetLook.set(pos.x, pos.y, pos.z);
  };
  CortexCinema.prototype.zoomToBest = function (on, leadPos) {
    this.zoomed = on;
    this.targetRadius = on ? 15 : 30;
    this.targetHeight = on ? 9 : 16;
    this.spin = on ? 0.0010 : 0.0022;
    if (on && leadPos) this.targetLook.set(leadPos.x, leadPos.y, leadPos.z);
  };
  CortexCinema.prototype.update = function (leadPos) {
    if (!this.zoomed) { // advance the shot cycle when not in zoom-to-best
      if (++this.shotT > this.shotLen) {
        this.shotT = 0; this.shot = (this.shot + 1) % this.shots.length;
        var s = this.shots[this.shot];
        this.targetRadius = s.radius; this.targetHeight = s.height; this.spin = s.spin;
      }
    }
    this.angle += this.spin;
    this.radius += (this.targetRadius - this.radius) * 0.045;
    this.height += (this.targetHeight - this.height) * 0.045;
    if (leadPos) this.frameLead(leadPos);
    this.look.lerp(this.targetLook, 0.06);
    var x = this.center.x + Math.cos(this.angle) * this.radius;
    var z = this.center.z + Math.sin(this.angle) * this.radius;
    var want = new THREE.Vector3(x, this.height, z);
    this.cam.position.lerp(want, 0.05);
    this.cam.lookAt(this.look);
  };
  root.CortexCinema = CortexCinema;
})(typeof window !== "undefined" ? window : globalThis);
