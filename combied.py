import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load data from Excel sheet
excel_file = 'E:\\t2\\actualvspredictcomplete.xlsx'
df = pd.read_excel(excel_file)

# Load the trained model
model = joblib.load("E:\\t2\\trained_model (1).pkl")


def predict_A_log_P(molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms):
    try:
        feature_vector = np.array([[molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms]])
        predicted_A_log_P = model.predict(feature_vector)
        return predicted_A_log_P[0][0]
    except ValueError:
        return None


def plot_linear_regression_graph(subset_size):
    df_subset = df.head(subset_size)
    predicted_log_p = df_subset['Predicted AlogP'].values.reshape(-1, 1)
    actual_log_p = df_subset['Actual AlogP'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(predicted_log_p, actual_log_p)
    predictions = model.predict(predicted_log_p)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Display linear regression equation and R-squared value
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    equation = f'Actual log P = {slope:.2f} * Predicted log P + {intercept:.2f}'
    r_squared = r2_score(actual_log_p, predictions)
    equation_text = f'Equation: {equation}\nR-squared: {r_squared:.2f}'
    ax.text(0.5, -0.15, equation_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

    # Plot linear regression graph
    ax.scatter(predicted_log_p, actual_log_p, label='Actual vs Predicted')
    ax.plot(predicted_log_p, predictions, color='red', label='Linear Regression Line')
    ax.set_xlabel('Predicted log P')
    ax.set_ylabel('Actual log P')
    ax.set_title(f'Linear Regression: Actual vs Predicted log P ({subset_size} Values)')
    ax.legend()
    ax.grid(True)

    # Pass the figure and axes to st.pyplot()
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="A log P Prediction", page_icon=":chart_with_upwards_trend:")

    st.title("A log P Prediction Web App")
    st.markdown("This web app predicts the A log P value for a molecule based on its features. Enter the details and click 'Predict A log P.'")

    # Section for A log P prediction
    st.header("A log P Prediction")

    col1, col2 = st.columns(2)

    with col1:
        molecular_weight = st.number_input("Molecular Weight", min_value=0.0)
        polar_surface_area = st.number_input("Polar Surface Area", min_value=0.0)
        hbd = st.number_input("Number of H-Bond Donors", min_value=0, step=1)
        hba = st.number_input("Number of H-Bond Acceptors", min_value=0, step=1)

    with col2:
        rotatable_bonds = st.number_input("Number of Rotatable Bonds", min_value=0, step=1)
        aromatic_rings = st.number_input("Number of Aromatic Rings", min_value=0, step=1)
        heavy_atoms = st.number_input("Number of Heavy Atoms", min_value=0.0)

    if st.button("Predict A log P"):
        result = predict_A_log_P(molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms)

        if result is not None:
            st.success(f"Predicted A log P: {result:.2f}")
        else:
            st.error("Invalid input. Please enter valid numeric values for all features.")

    st.markdown("---")  # Add a horizontal line to separate sections

    # Section for linear regression graph
    st.header("Linear Regression Graph")
    subset_size = st.slider("Select the number of values for the linear regression graph", min_value=10, max_value=len(df), value=100)
    plot_linear_regression_graph(subset_size)


if __name__ == "__main__":
    main()
