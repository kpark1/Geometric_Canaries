/- Minimal self-contained stand-in for the Aeneas Lean support library. -/
inductive Result (α : Type) where
  | ok (v : α)
  | fail
deriving Repr, DecidableEq

instance : Monad Result where
  pure := Result.ok
  bind x f := match x with | .ok v => f v | .fail => .fail

abbrev U32.max : Nat := 4294967295

/-- Machine-integer ops are fallible, as in the Aeneas translation:
    overflow/underflow surfaces as `.fail` (a Rust panic). -/
def U32.add (a b : Nat) : Result Nat :=
  if a + b ≤ U32.max then .ok (a + b) else .fail

def U32.sub (a b : Nat) : Result Nat :=
  if b ≤ a then .ok (a - b) else .fail

def U32.mul (a b : Nat) : Result Nat :=
  if a * b ≤ U32.max then .ok (a * b) else .fail

/-- Fallible indexing: out-of-bounds is a panic (`.fail`). -/
def List.index (l : List α) (i : Nat) : Result α :=
  match l[i]? with
  | some v => .ok v
  | none => .fail
