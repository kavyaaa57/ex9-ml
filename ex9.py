import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def lowess(x, y, f=2./3., iter=3):
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    for _ in range(iter):
        for i in range(n):
            weights = w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                           [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = np.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]
    return yest

@st.cache_data
def load_sample_data():
    sample_data = '''
    x,y
    1,2
    2,4
    3,6
    4,8
    5,10
    '''
    from io import StringIO
    return pd.read_csv(StringIO(sample_data))

def main():
    st.title("Non-Parametric Locally Weighted Regression (LOWESS)")

    # Upload data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.write("Using sample data.")
        data = load_sample_data()

    st.write(data.head())

    # Select columns for X and y
    columns = data.columns.tolist()
    x_col = st.selectbox("Select column for X", columns)
    y_col = st.selectbox("Select column for y", columns, index=1)

    # Fit LOWESS
    x = data[x_col].values
    y = data[y_col].values
    y_lowess = lowess(x, y)

    # Plot data and LOWESS curve
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data")
    ax.plot(x, y_lowess, 'r', label="LOWESS")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
