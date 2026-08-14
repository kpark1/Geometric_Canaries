def clamp (lo hi x : Nat) : Nat :=
  if x < lo then lo else if x > hi then hi else x
