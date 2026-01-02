
import torch
import sys

def test_empty_tensor():
    t = torch.zeros(0)
    print(f"Shape: {t.shape}")
    try:
        # Assuming clauses is usually 2D [num_clauses, max_len]
        # If it becomes empty, it might be [0, max_len] or just [0]
        # SATSolver init: self.clauses = torch.zeros(0) if empty
        # Assign: self.clauses = self.clauses[remain_mask]
        
        # If we start with [1, 2], [3]
        # Shape [2, 2]
        # Remove both -> Shape [0, 2]
        
        t2 = torch.zeros(0, 5)
        print(f"Shape t2: {t2.shape}")
        cnz = t2.count_nonzero(dim=1)
        print(f"CNZ: {cnz}")
        print(f"Any: {cnz.any()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_empty_tensor()
