#!/usr/bin/env python3
"""Generate insertion_depth.yaml for every s00XX patient in the SonoGym dataset.

The calculation reproduces the convention used by VertebraViewer:
- the centerline is obtained via PCA on the trajectory STL points;
- the direction is oriented toward +Y;
- half_length is the mean absolute projection onto the main axis;
- out = trajectory center + direction * half_length;
- in = first intersection with the vertebra along the +Y direction;
- drilling_depth = Euclidean distance between in and out.

YAML values are expressed in metres, as in the SonoGym benchmark.
Dependency: NumPy. PyYAML is not required.
"""

from __future__ import annotations

import argparse
import fnmatch
import struct
import sys
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(
    "/home/idsia/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/HumanModels/selected_dataset_stl"
)


def load_stl(path: Path) -> np.ndarray:
    """Load a binary or ASCII STL as an (N, 3, 3) triangle array."""
    data = path.read_bytes()

    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * triangle_count == len(data):
            dtype = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            )
            records = np.frombuffer(data, dtype=dtype, count=triangle_count, offset=84)
            return records["vertices"].astype(np.float64)

    vertices: list[list[float]] = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append([float(value) for value in fields[1:]])

    if not vertices or len(vertices) % 3:
        raise ValueError(f"Invalid or unsupported STL: {path}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def unique_points(triangles: np.ndarray) -> np.ndarray:
    """Return the equivalent of the unique points from pyvista.read(...).points."""
    return np.unique(triangles.reshape(-1, 3), axis=0)


def trajectory_parameters(
    trajectory_triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Reproduce VertebraViewer.get_traj_center_drct_radius()."""
    points = unique_points(trajectory_triangles)
    center = points.mean(axis=0)
    centered = points - center
    covariance = (centered.T @ centered) / (len(points) - 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    direction /= np.linalg.norm(direction)

    # SonoGym forces the trajectory direction toward +Y.
    if float(np.dot(direction, np.array([0.0, 1.0, 0.0]))) < 0.0:
        direction = -direction

    projections = centered @ direction
    half_length = float(np.mean(np.abs(projections)))
    final_point = center + direction * half_length
    return center, direction, half_length, final_point


def line_mesh_intersections(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    tolerance: float = 1e-7,
) -> np.ndarray:
    """Intersect origin + t*direction with the mesh (Moller-Trumbore)."""
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = v1 - v0, v2 - v0
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(determinant) > tolerance

    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    s = origin - v0
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.einsum("j,ij->i", direction, q)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    hit = valid & (u >= -tolerance) & (v >= -tolerance) & (u + v <= 1.0 + tolerance)

    # Merge duplicate hits produced on shared edges or vertices.
    values = np.sort(t[hit])
    if not len(values):
        return values
    groups: list[list[float]] = [[float(values[0])]]
    cluster_tolerance = tolerance * 100.0
    for value in values[1:]:
        value = float(value)
        if value - groups[-1][-1] <= cluster_tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return np.asarray([np.mean(group) for group in groups])


def find_entry_parameter(intersections: np.ndarray, final_t: float) -> float:
    """Find the entry of the bone segment containing the final point."""
    if len(intersections) < 2:
        raise RuntimeError(
            f"found {len(intersections)} vertebra intersections; at least 2 are required"
        )

    # For a closed mesh, ordered intersections form entry/exit pairs.
    usable_count = len(intersections) - (len(intersections) % 2)
    pairs = list(zip(intersections[:usable_count:2], intersections[1:usable_count:2]))

    for entry_t, cortex_exit_t in pairs:
        if entry_t <= final_t <= cortex_exit_t:
            return float(entry_t)

    # Some STL meshes are not perfectly closed. As a fallback, use the last
    # intersection encountered before the final point.
    preceding = intersections[intersections <= final_t]
    if len(preceding):
        return float(preceding[-1])
    raise RuntimeError("no entry intersection was found before the final point")


def analyze_pair(
    trajectory_path: Path,
    vertebra_path: Path,
    resolution_m_per_unit: float,
) -> dict[str, object]:
    trajectory = load_stl(trajectory_path)
    vertebra = load_stl(vertebra_path)
    center, direction, half_length, final_point = trajectory_parameters(trajectory)

    intersections = line_mesh_intersections(center, direction, vertebra)
    entry_t = find_entry_parameter(intersections, half_length)
    entry_point = center + direction * entry_t

    entry_m = entry_point * resolution_m_per_unit
    final_m = final_point * resolution_m_per_unit
    drilling_depth_m = float(np.linalg.norm(final_m - entry_m))

    return {
        "drilling_depth": drilling_depth_m,
        "in": entry_m,
        "out": final_m,
    }


def yaml_text(result: dict[str, object]) -> str:
    """Produce simple, portable YAML without requiring PyYAML."""
    point_in = np.asarray(result["in"], dtype=float)
    point_out = np.asarray(result["out"], dtype=float)
    return (
        "# Units: metres\n"
        f"drilling_depth: {float(result['drilling_depth']):.12g}\n"
        f"in: [{point_in[0]:.12g}, {point_in[1]:.12g}, {point_in[2]:.12g}]\n"
        f"out: [{point_out[0]:.12g}, {point_out[1]:.12g}, {point_out[2]:.12g}]\n"
    )


def matching_files(folder: Path, pattern: str) -> list[Path]:
    pattern = pattern.lower()
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and fnmatch.fnmatch(path.name.lower(), pattern)
    )


def select_pair(
    patient_folder: Path,
    side: str,
    trajectory_pattern: str | None,
) -> tuple[Path, Path]:
    if trajectory_pattern is None:
        trajectory_pattern = f"standard_{side}_traj_L4.stl"
    trajectories = matching_files(patient_folder, trajectory_pattern)

    if len(trajectories) != 1:
        names = ", ".join(path.name for path in trajectories) or "none"
        raise RuntimeError(
            f"expected 1 trajectory file matching {trajectory_pattern!r}, "
            f"found {len(trajectories)}: {names}"
        )

    trajectory = trajectories[0]

    # This batch is intentionally restricted to the L4 vertebra.
    expected_name = "vertebrae_L4.stl"
    vertebrae = sorted(
        path
        for path in patient_folder.iterdir()
        if path.is_file() and path.name.lower() == expected_name.lower()
    )
    if len(vertebrae) != 1:
        names = ", ".join(path.name for path in vertebrae) or "none"
        raise RuntimeError(
            f"expected exactly 1 {expected_name!r} file, "
            f"found {len(vertebrae)}: {names}"
        )
    return trajectory, vertebrae[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"selected_dataset_stl directory (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--patient-pattern",
        default="s00[0-9][0-9]",
        help="Patient directory pattern (default: s00[0-9][0-9])",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="right",
        help="Trajectory side to process (default: left)",
    )
    parser.add_argument(
        "--trajectory-pattern",
        default=None,
        help=(
            "Optional trajectory filename pattern. By default, uses "
            "standard_<side>_traj_L4.stl"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.0015,
        help="Metres per STL unit / label_res (default: 0.0015)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite insertion_depth.yaml when it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate and display results without writing files",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: dataset directory not found: {root}", file=sys.stderr)
        return 2

    patient_folders = sorted(
        folder
        for folder in root.iterdir()
        if folder.is_dir() and fnmatch.fnmatch(folder.name.lower(), args.patient_pattern.lower())
    )
    if not patient_folders:
        print(
            f"ERROR: no {args.patient_pattern!r} directory found in {root}",
            file=sys.stderr,
        )
        return 2

    completed = 0
    errors = 0
    skipped = 0

    for patient_folder in patient_folders:
        output_path = patient_folder / "insertion_depth.yaml"
        if output_path.exists() and not args.overwrite and not args.dry_run:
            print(f"SKIP  {patient_folder.name}: {output_path.name} esiste gia")
            skipped += 1
            continue

        try:
            trajectory_path, vertebra_path = select_pair(
                patient_folder,
                args.side,
                args.trajectory_pattern,
            )
            result = analyze_pair(trajectory_path, vertebra_path, args.resolution)
            content = yaml_text(result)

            if args.dry_run:
                print(
                    f"OK    {patient_folder.name}: "
                    f"{float(result['drilling_depth']) * 1000.0:.3f} mm "
                    f"[{trajectory_path.name} + {vertebra_path.name}]"
                )
            else:
                # Atomic write: the final file appears only after all content is ready.
                temporary_path = patient_folder / ".insertion_depth.yaml.tmp"
                temporary_path.write_text(content, encoding="utf-8")
                temporary_path.replace(output_path)
                print(
                    f"WRITE {patient_folder.name}/{output_path.name}: "
                    f"{float(result['drilling_depth']) * 1000.0:.3f} mm"
                )
            completed += 1
        except Exception as error:
            print(f"ERROR {patient_folder.name}: {error}", file=sys.stderr)
            errors += 1

    print(
        f"Completed: {completed} | Skipped: {skipped} | Errors: {errors} | "
        f"Total directories: {len(patient_folders)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
