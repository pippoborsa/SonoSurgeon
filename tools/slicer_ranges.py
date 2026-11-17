from __future__ import annotations

import argparse
import os


# =============================================================================
# ISAAC SIM DEVE ESSERE AVVIATO PRIMA DI IMPORTARE spinal_surgery
# =============================================================================

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Calculate USSlicer ranges for each patient."
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Non serve il viewer grafico
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# QUESTI IMPORT DEVONO ESSERE DOPO APP LAUNCHER
# =============================================================================

import torch
from ruamel.yaml import YAML

from spinal_surgery import ASSETS_DATA_DIR

# Metti qui lo stesso identico import usato nel task

from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer


# =============================================================================
# UNICO INPUT
# =============================================================================

ID_LIST = [
    "s0004",
    "s0006",
    "s0010",
    "s0012",
    "s0014",
    "s0015",
    "s0024",
    "s0028",
    "s0029",
    "s0030",
    "s0034",
    "s0038",
]


# =============================================================================
# CONFIGURAZIONE IDENTICA AL GUIDANCE ENVIRONMENT
# =============================================================================

DEVICE = "cpu"
OUTPUT_FILE = f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/patient_id/patient_us_ranges.yaml"

# Stessi margini usati nel secondo script
X_MARGIN = 20
Z_MARGIN = 30
INIT_MARGIN = 5


TARGET_ANATOMY = "vertebrae_L4"


def get_patient_files(patient_id: str) -> tuple[str, str]:
    patient_dir = os.path.join(
        ASSETS_DATA_DIR,
        "HumanModels",
        "selected_dataset_stl",
        patient_id,
    )

    vertebra_file = os.path.join(
        patient_dir,
        f"{TARGET_ANATOMY}.stl",
    )

    trajectory_file = os.path.join(
        patient_dir,
        f"standard_right_traj_{TARGET_ANATOMY[-2:]}.stl",
    )

    return vertebra_file, trajectory_file


def calculate_patient_ranges(patient_id: str) -> dict:
    vertebra_file, trajectory_file = get_patient_files(patient_id)

    vertebra_viewer = VertebraViewer(
        num_envs=1,
        n_human_types=1,
        vertebra_file_list=[vertebra_file],
        traj_file_list=[trajectory_file],
        if_vis=False,
        res=0.0015,
        device=DEVICE,
    )

    # =========================================================================
    # STESSO CALCOLO FATTO IN robot_US_guidance_Pink
    # =========================================================================

    vertebra_to_US_2d_pos = torch.tensor(
        [-30.0, 0.0],
        device=DEVICE,
        dtype=torch.float32,
    )

    vertebra_2d_pos = vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]

    goal_pos = vertebra_2d_pos + vertebra_to_US_2d_pos.unsqueeze(0)

    goal_angle_tensor = torch.full(
        (goal_pos.shape[0], 1),
        1.57,
        device=DEVICE,
        dtype=torch.float32,
    )

    goal_cmd_pose = torch.cat(
        [goal_pos, goal_angle_tensor],
        dim=1,
    )

    goal_x = goal_cmd_pose[0, 0].item()
    goal_z = goal_cmd_pose[0, 1].item()
    goal_angle = goal_cmd_pose[0, 2].item()

    patient_xz_range = [
        [
            goal_x - X_MARGIN,
            goal_x + X_MARGIN,
        ],
        [
            goal_z - Z_MARGIN,
            goal_z + Z_MARGIN,
        ],
    ]

    patient_xz_init_range = [
        [
            goal_x - X_MARGIN + INIT_MARGIN,
            goal_x + X_MARGIN - INIT_MARGIN,
        ],
        [
            goal_z - Z_MARGIN + INIT_MARGIN,
            goal_z + Z_MARGIN - INIT_MARGIN,
        ],
    ]

    return {
        "patient_xz_range": patient_xz_range,
        "patient_xz_init_range": patient_xz_init_range,
    }

def main() -> None:
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(
        mapping=2,
        sequence=4,
        offset=2,
    )

    for patient_id in ID_LIST:
        ranges = calculate_patient_ranges(patient_id)

        patient_dir = os.path.join(
            ASSETS_DATA_DIR,
            "HumanModels",
            "selected_dataset_stl",
            patient_id,
        )

        output_file = os.path.join(
            patient_dir,
            "patient_us_ranges.yaml",
        )

        output = ranges

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:
            yaml.dump(output, file)

        print(patient_id)
        print(
            "  patient_xz_range:",
            ranges["patient_xz_range"],
        )
        print(
            "  patient_xz_init_range:",
            ranges["patient_xz_init_range"],
        )
        print(
            "  salvato:",
            output_file,
        )


if __name__ == "__main__":
    main()