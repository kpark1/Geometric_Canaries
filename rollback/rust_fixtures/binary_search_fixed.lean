def binary_search_loop (v : List Nat) (x lo hi fuel : Nat) :
    Result (Option Nat) :=
  match fuel with
  | 0 => .fail
  | Nat.succ fuel =>
    if lo < hi then do
      let mid := lo + (hi - lo) / 2
      let vm ← v.index mid
      if vm = x then .ok (some mid)
      else if vm < x then binary_search_loop v x (mid + 1) hi fuel
      else binary_search_loop v x lo mid fuel
    else .ok none

def binary_search (v : List Nat) (x : Nat) : Result (Option Nat) :=
  binary_search_loop v x 0 v.length (v.length + 1)
