import numpy as np
import pandas as pd

def compute_tables(out_size, n_class, k_size, K_values=(4, 6, 8), J_values=(3, 4, 5)):
    """
    Compute four tables with:
      rows    -> K in K_values
      columns -> J in J_values

    Tables:
      T1 = (K+1)^J
      T2 = (K+1)^J * out_size^2
      T3 = (K+1)^J * out_size^2 * n_class
      T4 = sum_{j=1}^J (K+1)^j * K * k_size^2
    """

    K_vals = np.array(K_values)
    J_vals = np.array(J_values)

    # Initialize arrays
    T1 = np.zeros((len(K_vals), len(J_vals)), dtype=np.int64)
    T2 = np.zeros_like(T1, dtype=np.int64)
    T3 = np.zeros_like(T1, dtype=np.int64)
    T4 = np.zeros_like(T1, dtype=np.int64)

    for i, K in enumerate(K_vals):
        for j, J in enumerate(J_vals):
            base = (K + 1) ** J

            T1[i, j] = base
            T2[i, j] = base * out_size ** 2
            T3[i, j] = base * out_size ** 2 * n_class

            # geometric series sum
            T4[i, j] = ((K + 1) ** (J) -  1) * (k_size ** 2)

    # Wrap as DataFrames for readability
    index = [f"K={K}" for K in K_vals]
    columns = [f"J={J}" for J in J_vals]

    tables = {
        "Out Channel (K+1)^J": pd.DataFrame(T1, index=index, columns=columns),
        "Out Dim     (K+1)^J_out_size^2": pd.DataFrame(T2, index=index, columns=columns),
        "Linear Para (K+1)^J_out_size^2_n_class": pd.DataFrame(T3, index=index, columns=columns),
        "Tuned Para  sum_(K+1)^j_K_k_size^2": pd.DataFrame(T4, index=index, columns=columns),
    }

    return tables

if __name__ == "__main__":
    values = input("Please enter out_size, n_class, k_size separated by spaces:")
    out_size, n_class, k_size = map(int, values.split())
    tables = compute_tables(
        out_size=out_size,
        n_class=n_class,
        k_size=k_size
    )

    for name, table in tables.items():
        print(f"\n{name}")
        print(table)
