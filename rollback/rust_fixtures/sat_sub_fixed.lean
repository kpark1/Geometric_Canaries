def satSub (a b : Nat) : Result Nat :=
  if a ≥ b then U32.sub a b else .ok 0
