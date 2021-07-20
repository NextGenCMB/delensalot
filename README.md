# Lenscarf

Curved-sky iterative CMB lensing tools based on scarf


known 'issues'
* interpolation of the deflection vector gets poorer very very close to the poles not great (a couple of pixels)
  (could e.g. perform it exactly or almost exactly there instead, or use polar interpolation with scipy.interpolate.inter2pd,
  or just rescale the thing around the pole by ephi?)

