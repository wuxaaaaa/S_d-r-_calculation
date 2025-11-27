import streamlit as st
import numpy as np
import pandas as pd
import datetime
import tempfile
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.spatial import cKDTree
import io


def read_dump_file(filename: str) -> Tuple[np.ndarray, List[Tuple[float, float]], int]:
    """
    Read LAMMPS trajectory file and sort atoms by ID

    Args:
        filename: LAMMPS dump file path

    Returns:
        tuple: (atom data array, box boundary information)

    Raises:
        FileNotFoundError: When file does not exist
        ValueError: When file format is unexpected
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Trajectory file not found: {filename}")

    # Extract number of atoms from the file (4th line in LAMMPS dump format)
    try:
        n_atoms_line = lines[3].strip()
        n_atoms = int(n_atoms_line)
    except (IndexError, ValueError):
        raise ValueError("Cannot read number of atoms from file. Invalid format.")

    # Validate file has enough lines
    if len(lines) < 9 + n_atoms:
        raise ValueError(f"Insufficient file lines, expected at least {9 + n_atoms}, got {len(lines)}")

    # Extract box boundary information (lines 6-8 in LAMMPS dump format)
    box_lines = lines[5:8]
    box_bounds = []
    for line in box_lines:
        parts = line.strip().split()
        if len(parts) < 2:
            raise ValueError("Box boundary line format error")
        box_bounds.append((float(parts[0]), float(parts[1])))

    # Extract atom data
    atom_data = []
    for i in range(9, 9 + n_atoms):
        parts = lines[i].strip().split()
        if len(parts) < 5:
            raise ValueError(f"Line {i + 1} atom data format error")
        atom_data.append([int(parts[0]),  # Atom ID
                          int(parts[1]),  # Atom type
                          float(parts[2]),  # x coordinate
                          float(parts[3]),  # y coordinate
                          float(parts[4])  # z coordinate
                          ])

    atom_data = np.array(atom_data)
    # Sort by atom ID for consistency
    atom_data = atom_data[atom_data[:, 0].argsort()]

    print(f"Read {n_atoms} atoms from file")
    return atom_data, box_bounds, n_atoms


def create_periodic_images(
        atoms: np.ndarray,
        bounds: List[Tuple[float, float]],
        buffer_distance: float = 10.5
) -> np.ndarray:
    """
    Create periodic images for atoms near boundaries

    Args:
        atoms: Atom data array with shape (n_atoms, 5)
        bounds: Box boundaries list [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        buffer_distance: Boundary buffer distance

    Returns:
        np.ndarray: Extended array containing original atoms and image atoms

    """
    x_bounds, y_bounds, z_bounds = bounds
    dx, dy, dz = (
        x_bounds[1] - x_bounds[0],
        y_bounds[1] - y_bounds[0],
        z_bounds[1] - z_bounds[0]
    )

    # # Define boundary regions
    x_near_boundary = (x_bounds[0] + buffer_distance, x_bounds[1] - buffer_distance)
    y_near_boundary = (y_bounds[0] + buffer_distance, y_bounds[1] - buffer_distance)
    z_near_boundary = (z_bounds[0] + buffer_distance, z_bounds[1] - buffer_distance)

    positions = atoms[:, 2:5]  # Extract coordinates (x, y, z)

    # Identify atoms near boundaries
    near_x_boundary = (positions[:, 0] < x_near_boundary[0]) | (positions[:, 0] > x_near_boundary[1])
    near_y_boundary = (positions[:, 1] < y_near_boundary[0]) | (positions[:, 1] > y_near_boundary[1])
    near_z_boundary = (positions[:, 2] < z_near_boundary[0]) | (positions[:, 2] > z_near_boundary[1])

    near_boundary_mask = near_x_boundary | near_y_boundary | near_z_boundary
    boundary_atoms = atoms[near_boundary_mask]

    # Return original array if no boundary atoms
    if len(boundary_atoms) == 0:
        return atoms

    # Define all possible image translation vectors
    translations = _generate_translation_vectors(dx, dy, dz)

    # Generate image atoms
    all_images = []
    for dx_trans, dy_trans, dz_trans in translations:
        images = boundary_atoms.copy()
        images[:, 2] += dx_trans  # x coordinate
        images[:, 3] += dy_trans  # y coordinate
        images[:, 4] += dz_trans  # z coordinate
        all_images.append(images)

    # Combine all atoms
    if all_images:
        all_images = np.vstack(all_images)
        extended_atoms = np.vstack([atoms, all_images])
    else:
        extended_atoms = atoms

    return extended_atoms


def _generate_translation_vectors(
        dx: float,
        dy: float,
        dz: float
) -> List[Tuple[float, float, float]]:
    """
    Generate all possible periodic image translation vectors

    Args:
        dx, dy, dz: Box lengths in three directions

    Returns:
        list: List of translation vectors [(dx, dy, dz), ...]
    """
    translations = []

    # Single direction images
    single_direction = [
        (-dx, 0, 0), (dx, 0, 0),
        (0, -dy, 0), (0, dy, 0),
        (0, 0, -dz), (0, 0, dz)
    ]
    translations.extend(single_direction)

    # Double direction images
    double_direction = [
        (-dx, -dy, 0), (-dx, dy, 0), (dx, -dy, 0), (dx, dy, 0),
        (-dx, 0, -dz), (-dx, 0, dz), (dx, 0, -dz), (dx, 0, dz),
        (0, -dy, -dz), (0, -dy, dz), (0, dy, -dz), (0, dy, dz)
    ]
    translations.extend(double_direction)

    # Triple direction images
    triple_direction = [
        (-dx, -dy, -dz), (-dx, -dy, dz), (-dx, dy, -dz), (-dx, dy, dz),
        (dx, -dy, -dz), (dx, -dy, dz), (dx, dy, -dz), (dx, dy, dz)
    ]
    translations.extend(triple_direction)

    return translations


def analyze_neighbors_vectorized_sorted(
        original_atoms: np.ndarray,
        extended_atoms: np.ndarray,
        r_min: float = 2,
        r_max: float = 10,
        n_bins: int = 40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze neighbor distribution around each atom, ensuring results are sorted by atom ID

    Args:
        original_atoms: Original atom data
        extended_atoms: Extended data including image atoms
        r_min: Minimum analysis distance
        r_max: Maximum analysis distance
        n_bins: Number of distance bins

    Returns:
        tuple: (Total vector modulus, Type 1 vector modulus, Type 2 vector modulus, Atom IDs)
    """

    # Extract coordinates and IDs
    original_ids = original_atoms[:, 0].astype(int)
    original_positions = original_atoms[:, 2:5].astype(np.float64)
    extended_positions = extended_atoms[:, 2:5].astype(np.float64)
    extended_types = extended_atoms[:, 1].astype(int)

    # Build KD-tree for fast neighbor search
    kd_tree = cKDTree(extended_positions)

    # Set up distance bins
    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    # Process each atom in order of atom ID
    vector_sum = np.zeros((len(original_atoms), n_bins, 3))
    vector_sum_type1 = np.zeros((len(original_atoms), n_bins, 3))
    vector_sum_type2 = np.zeros((len(original_atoms), n_bins, 3))
    scalar_sum = np.zeros((len(original_atoms), n_bins))
    scalar_sum_type1 = np.zeros((len(original_atoms), n_bins))
    scalar_sum_type2 = np.zeros((len(original_atoms), n_bins))

    # 按原子ID顺序处理每个原子
    for atom_index in range(len(original_atoms)):
        center_position = original_positions[atom_index]
        center_id = original_ids[atom_index]

        # Query all neighbors within radius
        neighbor_indices = kd_tree.query_ball_point(center_position, r=r_max)

        if not neighbor_indices:
            continue

        neighbor_indices = np.array(neighbor_indices)
        # Exclude self (by comparing atom ID)
        non_self_mask = extended_atoms[neighbor_indices, 0] != center_id
        neighbor_indices = neighbor_indices[non_self_mask]

        if len(neighbor_indices) == 0:
            continue

        # Calculate relative vectors and distances
        neighbor_vectors = extended_positions[neighbor_indices] - center_position
        neighbor_distances = np.linalg.norm(neighbor_vectors, axis=1)

        # Filter neighbors within valid distance range
        valid_distance_mask = (neighbor_distances >= r_min) & (neighbor_distances <= r_max)
        if not np.any(valid_distance_mask):
            continue

        valid_vectors = neighbor_vectors[valid_distance_mask]
        valid_distances = neighbor_distances[valid_distance_mask]
        valid_indices = neighbor_indices[valid_distance_mask]
        valid_types = extended_types[valid_indices]

        # Assign to distance bins
        bin_indices = np.floor((valid_distances - r_min) / bin_width).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure valid indices

        # Accumulate statistics by bin
        _accumulate_bin_statistics(
            atom_index, bin_indices, valid_vectors, valid_distances, valid_types,
            vector_sum, vector_sum_type1, vector_sum_type2,
            scalar_sum, scalar_sum_type1, scalar_sum_type2,
            n_bins
        )

    # Calculate vector modulus
    vector_modulus = np.linalg.norm(vector_sum, axis=2)
    vector_modulus_type1 = np.linalg.norm(vector_sum_type1, axis=2)
    vector_modulus_type2 = np.linalg.norm(vector_sum_type2, axis=2)

    return vector_modulus, vector_modulus_type1, vector_modulus_type2, original_ids


def _accumulate_bin_statistics(
        atom_index: int,
        bin_indices: np.ndarray,
        vectors: np.ndarray,
        distances: np.ndarray,
        types: np.ndarray,
        vector_sum: np.ndarray,
        vector_sum_type1: np.ndarray,
        vector_sum_type2: np.ndarray,
        scalar_sum: np.ndarray,
        scalar_sum_type1: np.ndarray,
        scalar_sum_type2: np.ndarray,
        n_bins: int
) -> None:
    """
    Accumulate statistics by bin for neighbor data of a specific atom

    Args:
        atom_index: Current atom index
        bin_indices: Bin index for each neighbor
        vectors: Neighbor relative vectors
        distances: Neighbor distances
        types: Neighbor atom types
        vector_sum, vector_sum_type1, vector_sum_type2: Vector accumulation arrays
        scalar_sum, scalar_sum_type1, scalar_sum_type2: Scalar accumulation arrays
        n_bins: Number of bins
    """
    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        if not np.any(bin_mask):
            continue

        # Statistics for all atom types
        vector_sum[atom_index, bin_idx] += np.sum(vectors[bin_mask], axis=0)
        scalar_sum[atom_index, bin_idx] += np.sum(distances[bin_mask])

        # Statistics for type 1 atoms
        type1_mask = bin_mask & (types == 1)
        if np.any(type1_mask):
            vector_sum_type1[atom_index, bin_idx] += np.sum(vectors[type1_mask], axis=0)
            scalar_sum_type1[atom_index, bin_idx] += np.sum(distances[type1_mask])

        # Statistics for type 2 atoms
        type2_mask = bin_mask & (types == 2)
        if np.any(type2_mask):
            vector_sum_type2[atom_index, bin_idx] += np.sum(vectors[type2_mask], axis=0)
            scalar_sum_type2[atom_index, bin_idx] += np.sum(distances[type2_mask])

@st.cache_data
def run_analysis(uploaded_file, r_min, r_max, n_bins):
    dump_bytes = uploaded_file.read()
    temp_path = "uploaded_dump.dump"
    with open(temp_path, "wb") as f:
        f.write(dump_bytes)

    atom_data, box_bounds, n_atoms = read_dump_file(temp_path)
    x_length = box_bounds[0][1] - box_bounds[0][0]
    y_length = box_bounds[1][1] - box_bounds[1][0]
    z_length = box_bounds[2][1] - box_bounds[2][0]
    volume = x_length * y_length * z_length
    extended_data = create_periodic_images(atom_data, box_bounds)
    mod_vector, mod_vector_type1, mod_vector_type2, atom_ids = \
        analyze_neighbors_vectorized_sorted(
            atom_data, extended_data,
            r_min=r_min, r_max=r_max, n_bins=n_bins
        )
    a = volume / n_atoms
    scale_factor = a ** (1 / 3)
    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    r_list = (bin_edges[:-1] + bin_edges[1:]) / 2
    denominators = 4 * np.pi * (r_list ** 2)
    normalized_vector_sum = mod_vector/ denominators * scale_factor

    output_mod = io.StringIO()
    output_normalized = io.StringIO()
    np.savetxt(output_mod, mod_vector, fmt="%.6f")
    np.savetxt(output_normalized, normalized_vector_sum, fmt="%.6f")
    output_mod.seek(0)
    output_normalized.seek(0)
    return output_mod, output_normalized, normalized_vector_sum, r_list


def calculate_percentile_curves(frequency_data, sd_data, percentile_low=10, percentile_high=90):
    low_threshold = np.percentile(frequency_data, percentile_low)
    high_threshold = np.percentile(frequency_data, percentile_high)
    low_indices = frequency_data <= low_threshold
    high_indices = frequency_data >= high_threshold
    low_percentile_avg = np.mean(sd_data[low_indices], axis=0)
    high_percentile_avg = np.mean(sd_data[high_indices], axis=0)

    return low_percentile_avg, high_percentile_avg, low_indices, high_indices

# ---------------- Streamlit 网页 -----------------
st.title("Calculation of Sd(r)")
st.write("Upload LAMMPS trajectory file (1 fram; format:id type x y z)")

uploaded_file = st.file_uploader("Upload LAMMPS trajectory file", type=["dump", "txt"])

st.subheader("R setting")
r_min = st.number_input("R_MIN", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
r_max = st.number_input("R_MAX", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
n_bins = st.number_input("N_BINS", min_value=5, max_value=200, value=40, step=1)


if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "normalized_vector_sum" not in st.session_state:
    st.session_state.normalized_vector_sum = None
if "r_list" not in st.session_state:
    st.session_state.r_list = None
if "output_mod_str" not in st.session_state:
    st.session_state.output_mod_str = None
if "output_normalized_str" not in st.session_state:
    st.session_state.output_normalized_str = None


if uploaded_file and st.button("Start Analysis", key="start_analysis"):
    st.write("Calculating, please wait...")
    with st.spinner("Running analysis..."):
        try:
            output_mod, output_normalized, normalized_vector_sum, r_list = run_analysis(uploaded_file, r_min, r_max, n_bins)
        except Exception as e:
            st.error(f"Run analysis failed: {e}")
            raise


    st.session_state.analysis_done = True
    st.session_state.normalized_vector_sum = normalized_vector_sum
    st.session_state.r_list = r_list
    st.session_state.output_mod_str = output_mod.getvalue()
    st.session_state.output_normalized_str = output_normalized.getvalue()


if st.session_state.analysis_done:
    st.success("Done!")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download vector sum",
            data=st.session_state.output_mod_str,
            file_name=f"vector_sum_bins{n_bins}.txt",
            mime="text/plain",
            key="download_vector_sum"
        )

    with col2:
        st.download_button(
            label="Download Sd(r)",
            data=st.session_state.output_normalized_str,
            file_name=f"Sd(r)_bins{n_bins}.txt",
            mime="text/plain",
            key="download_sdr"
        )

    st.subheader("Further Analysis - Plot liquid-like and solid-like Sd(r)")


    analysis_option = st.radio(
        "Choose label prediction method:",
        ["No further analysis", "Upload frequency file", "Use ML model prediction"],
        key="analysis_option_radio"
    )

    # ---------------- Upload frequency file branch ----------------
    if analysis_option == "Upload frequency file":
        frequency_file = st.file_uploader("Upload atom frequency file", type=["txt", "dat"], key="frequency_file")

        if frequency_file is not None and st.button("Calculate Liquid-Solid Curves", key="calc_liquid_solid"):
            if st.session_state.normalized_vector_sum is None:
                st.error("No Sd(r) data found. Please run 'Start Analysis' first.")
            else:
                try:
                    # load frequency data
                    frequency_file.seek(0)
                    frequency_data = np.loadtxt(frequency_file)
                except Exception as e:
                    st.error(f"Failed to read frequency file: {e}")
                    frequency_data = None

                if frequency_data is not None:
                    liquid_curve, solid_curve, liquid_indices, solid_indices = calculate_percentile_curves(
                        frequency_data, st.session_state.normalized_vector_sum
                    )
                    output_liquid = io.StringIO()
                    output_solid = io.StringIO()
                    np.savetxt(output_liquid, liquid_curve, fmt="%.6f")
                    np.savetxt(output_solid, solid_curve, fmt="%.6f")
                    output_liquid.seek(0)
                    output_solid.seek(0)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(st.session_state.r_list, liquid_curve, linewidth=2, label='Liquid-like (lowest 10%)')
                    ax.plot(st.session_state.r_list, solid_curve, linewidth=2, label='Solid-like (highest 10%)')
                    ax.set_xlabel('r (Å)')
                    ax.set_ylabel('Sd(r)')
                    ax.set_title('Comparison of Liquid-like and Solid-like Sd(r)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    col3, col4 = st.columns(2)
                    with col3:
                        st.download_button(
                            label="Download Liquid-like Sd(r)",
                            data=output_liquid.getvalue(),
                            file_name="liquid_like_sd_curve.txt",
                            mime="text/plain",
                            key="dl_liquid"
                        )

                    with col4:
                        st.download_button(
                            label="Download Solid-like Sd(r)",
                            data=output_solid.getvalue(),
                            file_name="solid_like_sd_curve.txt",
                            mime="text/plain",
                            key="dl_solid"
                        )

    # ---------------- Use ML model prediction branch ----------------
    elif analysis_option == "Use ML model prediction":
        uploaded_model = st.file_uploader(
            "Upload XGBoost model file (.json / .bin / .model)",
            type=["json", "bin", "model"],
            key="uploaded_xgb_model"
        )

        if st.button("Predict Frequencies and Calculate Curves", key="predict_ml"):
            if st.session_state.normalized_vector_sum is None:
                st.error("No Sd(r) data found. Please run 'Start Analysis' first.")
            else:
                st.write("Predicting frequencies and calculating curves...")
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_model.read())
                        tmp_path = tmp.name
                    model = xgb.Booster()
                    model.load_model(tmp_path)

                    dmatrix = xgb.DMatrix(st.session_state.normalized_vector_sum)
                    frequency_predictions = model.predict(dmatrix)
                except Exception as e:
                    st.error(f"ML prediction failed: {e}")
                    frequency_predictions = None

                if frequency_predictions is not None:
                    liquid_curve, solid_curve, liquid_indices, solid_indices = calculate_percentile_curves(
                        frequency_predictions, st.session_state.normalized_vector_sum
                    )


                    output_liquid = io.StringIO()
                    output_solid = io.StringIO()
                    output_frequencies = io.StringIO()

                    np.savetxt(output_liquid, liquid_curve, fmt="%.6f")
                    np.savetxt(output_solid, solid_curve, fmt="%.6f")
                    np.savetxt(output_frequencies, frequency_predictions, fmt="%.6f")

                    output_liquid.seek(0)
                    output_solid.seek(0)
                    output_frequencies.seek(0)


                    st.write(f"Number of liquid-like atoms (lowest 10% frequency): {np.sum(liquid_indices)}")
                    st.write(f"Number of solid-like atoms (highest 10% frequency): {np.sum(solid_indices)}")


                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(st.session_state.r_list, liquid_curve, linewidth=2, label='Liquid-like (lowest 10%)')
                    ax.plot(st.session_state.r_list, solid_curve, linewidth=2, label='Solid-like (highest 10%)')
                    ax.set_xlabel('r (Å)')
                    ax.set_ylabel('Sd(r)')
                    ax.set_title('Comparison of Liquid-like and Solid-like Sd(r)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)


                    col3, col4, col5 = st.columns(3)

                    with col3:
                        st.download_button(
                            label="Download Liquid-like Sd(r)",
                            data=output_liquid.getvalue(),
                            file_name="liquid_like_sd_curve.txt",
                            mime="text/plain",
                            key="dl_liquid_ml"
                        )

                    with col4:
                        st.download_button(
                            label="Download Solid-like Sd(r)",
                            data=output_solid.getvalue(),
                            file_name="solid_like_sd_curve.txt",
                            mime="text/plain",
                            key="dl_solid_ml"
                        )

                    with col5:
                        st.download_button(
                            label="Download Predicted Frequencies",
                            data=output_frequencies.getvalue(),
                            file_name="predicted_frequencies.txt",
                            mime="text/plain",
                            key="dl_freqs"
                        )
