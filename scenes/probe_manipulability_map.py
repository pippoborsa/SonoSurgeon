"""Random probe exploration with a stationary drill, using independent Pink IK.

Run with Isaac Lab's Python; the paired YAML defaults to this script's directory.
Outputs: results/<patient>_<timestamp>/{mean_map,covariance_map}.{npz,png}, samples.npz.
Map axes are recentered measured probe x,z (voxels) and angle (rad).
Raw patient-frame coordinates are preserved; mm axes are also exported.
Probe tracking error is diagnostic only; acceptance checks the drill and map range.
Aggregate measured cell means across envs, then interpolate the mean map.
Covariance map = pointwise sample variance across measured env maps only
(diagonal of spatial covariance), not covariance between different cells.
"""
from pathlib import Path
import argparse
import json
import logging
import math
from datetime import datetime
import numpy as np


class JointLimitFilter(logging.Filter):
    """Same joint-limit warning filter as the training script."""
    def filter(self, record):
        msg = record.getMessage()
        return not ("is out of limits" in msg and "Value" in msg)


def reflect(values, lower, upper):
    width = upper - lower
    phase = np.mod(values - lower, 2 * width)
    return lower + np.minimum(phase, 2 * width - phase)


def manipulability(probe_jacobian, drill_jacobian=None):
    """Product of singular values: stable Yoshikawa, without a fake positive floor."""
    jac = probe_jacobian
    if drill_jacobian is not None:
        # Nullspace projection is optional; ordinary Yoshikawa remains the default.
        jac = jac @ (np.eye(jac.shape[1]) - np.linalg.pinv(drill_jacobian) @ drill_jacobian)
    return float(np.prod(np.linalg.svd(jac, compute_uv=False)))


def build_maps(commands, values, accepted, bounds, settings):
    """Average measured env cell means equally, then interpolate only the mean."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError, cKDTree

    bins = np.asarray(settings['bins'], dtype=int)
    lo, hi = np.asarray(bounds, dtype=float)
    edges = [np.linspace(lo[d], hi[d], bins[d] + 1) for d in range(3)]
    axes = [(a[:-1] + a[1:]) / 2 for a in edges]
    active = bins > 1
    dimensions = int(active.sum())
    if bins[2] == 1 and settings.get('fixed_angle_rad') is not None:
        axes[2] = np.array([settings['fixed_angle_rad']])
    shape = tuple(bins)
    cells = np.stack(np.meshgrid(*[np.arange(b) for b in bins], indexing='ij'), -1)
    queries = cells.reshape(-1, 3)[:, active]
    n_envs = commands.shape[1]
    per_env = np.full((n_envs, *shape), np.nan)
    counts = np.zeros((n_envs, *shape), dtype=np.int64)
    for env in range(n_envs):
        valid = accepted[:, env] & np.isfinite(values[:, env])
        valid &= np.all((commands[:, env, active] >= lo[active]) & (commands[:, env, active] <= hi[active]), axis=1)
        xyz = commands[valid, env]
        idx = np.minimum(((xyz - lo) / (hi - lo) * bins).astype(int), bins - 1)
        idx[:, ~active] = 0
        sums = np.zeros(shape)
        np.add.at(sums, tuple(idx.T), values[valid, env])
        np.add.at(counts[env], tuple(idx.T), 1)
        observed = counts[env] >= settings['min_cell_samples']
        per_env[env, observed] = sums[observed] / counts[env, observed]
    supported = np.isfinite(per_env)
    contributors = supported.sum(0)
    mean = np.divide(np.nansum(per_env, axis=0), contributors,
                     out=np.full(shape, np.nan), where=contributors > 0)
    ss = np.nansum((per_env - mean) ** 2, axis=0)
    variance = np.divide(ss, contributors - 1, out=np.full(shape, np.nan),
                         where=contributors > 1)
    observed = contributors > 0
    points = cells[observed][:, active]
    if len(points) >= dimensions + 1 and np.linalg.matrix_rank(points - points.mean(0)) == dimensions:
        try:
            interp = LinearNDInterpolator(points, mean[observed])(queries)
            distance = cKDTree(points).query(queries)[0]
            interp[distance > settings['max_interpolation_distance_cells']] = np.nan
            mean[~observed] = interp.reshape(shape)[~observed]
        except QhullError:
            pass
    return axes, mean, variance, per_env, counts, contributors


def save_results(folder, records, bounds, cfg, diagnostics=None):
    positions, values, errors, accepted = [np.asarray(v) for v in records]
    axes, mean, variance, per_env, counts, contributors = build_maps(
        positions, values, accepted, bounds, cfg['mapping'])
    # Display/export translation only: control, acceptance and binning stay absolute.
    absolute_bounds = np.asarray(bounds, dtype=float)
    center = np.asarray(cfg['mapping'].get('center_xz_voxels',
                                          absolute_bounds.mean(axis=0)[:2]), dtype=float)
    offset = center - absolute_bounds.mean(axis=0)[:2]
    display_bounds = absolute_bounds.copy()
    display_bounds[:, :2] += offset
    x, z = axes[0] + offset[0], axes[1] + offset[1]
    mm_per_voxel = float(cfg['patient']['label_res']) * 1000
    cfg = dict(cfg, aggregation_method='equal_env_cell_means_then_interpolate',
               variance_method='sample_variance_of_observed_env_cell_means_ddof1',
               map_coordinate_transform='absolute_xz - range_midpoint_xz + center_xz_voxels',
               map_center_xz_voxels=center.tolist(), mm_per_voxel=mm_per_voxel)
    common = dict(x=x, z=z, angle=axes[2], bounds=display_bounds,
                  x_absolute=axes[0], z_absolute=axes[1], bounds_absolute=absolute_bounds,
                  x_mm=x * mm_per_voxel, z_mm=z * mm_per_voxel,
                  bounds_xz_mm=display_bounds[:, :2] * mm_per_voxel,
                  coordinate_offset_xz_voxels=offset,
                  contributors=contributors, observed_counts=counts,
                  observed_mask=contributors > 0,
                  interpolated_mask=np.isfinite(mean) & (contributors == 0),
                  metadata=json.dumps(cfg))
    np.savez_compressed(folder / 'mean_map.npz', mean=mean, per_env_maps=per_env, **common)
    np.savez_compressed(folder / 'covariance_map.npz', variance=variance, ddof=1, **common)
    extra = {} if diagnostics is None else {key: np.asarray(value) for key, value in diagnostics.items()}
    np.savez_compressed(folder / 'samples.npz', positions=positions, manipulability=values,
                        tracking_errors=errors, accepted=accepted,
                        **extra,
                        time_s=(np.arange(len(values)) + 1) * cfg['sim']['dt'],
                        error_columns=['probe_position_m', 'probe_angle_deg',
                                       'drill_position_m', 'drill_angle_deg'])
    if cfg['mapping']['png']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plot_bounds = display_bounds[:, :2] * mm_per_voxel
        for name, data in [('mean_map', mean), ('covariance_map', variance)]:
            fig, panels = plt.subplots(1, len(axes[2]), figsize=(4 * len(axes[2]), 4),
                                       squeeze=False, constrained_layout=True)
            finite = data[np.isfinite(data)]
            vmax = max(float(finite.max()), 1e-15) if finite.size else 1.0
            for k, ax in enumerate(panels[0]):
                image = ax.imshow(data[:, :, k].T, origin='lower', aspect='auto',
                                  extent=[plot_bounds[0, 0], plot_bounds[1, 0],
                                          plot_bounds[0, 1], plot_bounds[1, 1]],
                                  interpolation='nearest',
                                  vmin=0, vmax=vmax, cmap='viridis')
                ax.set(title=f'Angle = {axes[2][k]:.3f} rad',
                       xlabel='x shift [mm]', ylabel='z shift [mm]')
                if cfg['mapping'].get('plot_xlim_mm') is not None:
                    ax.set_xlim(cfg['mapping']['plot_xlim_mm'])
                if cfg['mapping'].get('plot_ylim_mm') is not None:
                    ax.set_ylim(cfg['mapping']['plot_ylim_mm'])
                if cfg['mapping'].get('plot_xticks_mm') is not None:
                    ax.set_xticks(cfg['mapping']['plot_xticks_mm'])
                if cfg['mapping'].get('plot_yticks_mm') is not None:
                    ax.set_yticks(cfg['mapping']['plot_yticks_mm'])
            fig.colorbar(image, ax=panels.ravel().tolist(), label=name.replace('_', ' '))
            fig.savefig(folder / f'{name}.png', dpi=180)
            plt.close(fig)
    print(f'Saved {folder}; accepted {accepted.sum()}/{accepted.size} samples; '
          f'mean coverage {np.isfinite(mean).mean():.1%}, variance coverage '
          f'{np.isfinite(variance).mean():.1%}.')


def run(args, cfg, app):
    import torch
    import nibabel as nib
    import pinocchio as pin
    from pink.configuration import Configuration
    from pink.tasks import FrameTask
    from pink.solve_ik import solve_ik
    from ruamel.yaml import YAML
    from scipy.spatial.transform import Rotation
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import (combine_frame_transforms as combine,
                                    subtract_frame_transforms as relative,
                                    matrix_from_quat, quat_from_matrix)
    from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
    from spinal_surgery.assets.unitreeG1 import G1_TOOLS_SURGERY_CFG
    from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
    from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
    from spinal_surgery.lab.kinematics.gt_motion_generator import GTDiscreteMotionGenerator

    joint_limit_filter = JointLimitFilter()
    root_logger = logging.getLogger()
    root_logger.addFilter(joint_limit_filter)
    # Handler filters also catch records propagated from Pink child loggers.
    for handler in root_logger.handlers:
        handler.addFilter(joint_limit_filter)
    logging.getLogger('pink.configuration').addFilter(joint_limit_filter)

    def read_yaml(path):
        with open(path, encoding='utf-8') as f:
            return YAML(typ='safe').load(f)

    def quaternion(sequence, angles):
        return tuple(Rotation.from_euler(sequence, angles, degrees=True).as_quat()[[3, 0, 1, 2]])

    def cpu(t):
        return t.detach().cpu().numpy()

    rc, pc, sc, bc = cfg['robot'], cfg['patient'], cfg['sim'], cfg['bed']
    if rc['type'] != 'g1':
        raise ValueError('This script implements the supplied G1 training configuration.')
    asset = Path(ASSETS_DATA_DIR)
    patient_dir = asset / 'HumanModels/selected_dataset_stl' / pc['id']
    bounds = np.asarray(sc['patient_xz_range'], dtype=float)
    initial = np.asarray(sc['patient_xz_init_range'], dtype=float)
    if sc['centered_ranges']:
        ranges = read_yaml(patient_dir / 'patient_us_ranges.yaml')
        bounds[:, :2] = np.asarray(ranges['patient_xz_range'])[:2, :2].T
        initial[:, :2] = np.asarray(ranges['patient_xz_init_range'])[:2, :2].T
    initial[0] = np.maximum(initial[0], bounds[0])
    initial[1] = np.minimum(initial[1], bounds[1])
    if np.any(bounds[1] <= bounds[0]) or np.any(initial[1] < initial[0]):
        raise ValueError('Invalid centered ranges or non-overlapping initial range.')
    cfg['resolved_range'] = bounds.tolist()
    fixed_angle = cfg['probe'].get('fixed_angle_rad')
    if fixed_angle is not None:
        if not bounds[0, 2] <= fixed_angle <= bounds[1, 2]:
            raise ValueError('probe.fixed_angle_rad must be inside the configured angular range.')
        cfg['mapping']['bins'][2] = 1
        cfg['mapping']['fixed_angle_rad'] = fixed_angle
    robot_cfg = G1_TOOLS_SURGERY_CFG.copy()
    robot_cfg.prim_path = '/World/envs/env_.*/Robot_US'
    robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=tuple(rc['pos']), rot=quaternion('z', rc['yaw']),
        joint_pos=robot_cfg.init_state.joint_pos, joint_vel=robot_cfg.init_state.joint_vel)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = robot_cfg
        ground = AssetBaseCfg(prim_path='/World/ground', spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path='/World/light', spawn=sim_utils.DomeLightCfg(intensity=3000))
        bed = AssetBaseCfg(
            prim_path='/World/envs/env_.*/Bed',
            spawn=sim_utils.UsdFileCfg(usd_path=str(asset / 'MedicalBed/usd_colored/hospital_bed.usd'),
                                       scale=(bc['scale'],) * 3),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(bc['pos']), rot=quaternion('xyz', bc['euler_xyz'])))
        human = RigidObjectCfg(
            prim_path='/World/envs/env_.*/Human',
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset / 'HumanModels/selected_dataset_body_from_urdf' / pc['id'] /
                             'combined_wrapwrap/combined_wrapwrap.usd'),
                scale=(pc['label_res'],) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pc['pos']), rot=quaternion('yxz', pc['euler_yxz'])))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=sc['dt'], device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=sc['num_envs'], env_spacing=sc['env_spacing'], replicate_physics=False))
    sim.reset()
    scene.update(sc['dt'])
    robot, human = scene['robot'], scene['human']
    n, device, dt = sc['num_envs'], sim.device, sc['dt']
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    probe_frame, drill_frame = 'left_wrist_yaw_link', 'right_wrist_yaw_link'
    entities = []
    for side, frame in [('left', probe_frame), ('right', drill_frame)]:
        e = SceneEntityCfg('robot', joint_names=[rf'(torso_joint|waist_(pitch|roll|yaw)_joint|{side}_(shoulder|elbow|wrist)_.*)'], body_names=[frame])
        e.resolve(scene)
        entities.append(e)
    controlled = sorted(set(entities[0].joint_ids + entities[1].joint_ids))
    names = list(robot.data.joint_names)
    full = pin.buildModelFromUrdf(str(asset / 'unitree/robots/urdf/g1/g1_body29_hand14.urdf'))
    reference = pin.neutral(full)
    name_index = {name: i for i, name in enumerate(names)}
    q0 = cpu(robot.data.joint_pos[0])
    for j, name in enumerate(full.names):
        if name in name_index and full.joints[j].nq == 1:
            reference[full.joints[j].idx_q] = q0[name_index[name]]
    ctrl_names = {names[i] for i in controlled}
    locked = [j for j in range(1, full.njoints) if full.names[j] not in ctrl_names]
    model = pin.buildReducedModel(full, locked, reference)
    q_idx, v_idx = [], []
    for i in controlled:
        jid = model.getJointId(names[i])
        if jid >= model.njoints or model.joints[jid].nq != 1:
            raise ValueError(f'Unmapped controlled joint: {names[i]}')
        q_idx.append(model.joints[jid].idx_q)
        v_idx.append(model.joints[jid].idx_v)
    probe_id = model.getFrameId(probe_frame)
    wrist_id = model.getFrameId(drill_frame)
    wrist_frame = model.frames[wrist_id]
    tool_transform = pin.SE3(
        Rotation.from_euler('YXZ', cfg['drill']['wrist_to_tip_euler_YXZ'], degrees=True).as_matrix(),
        np.asarray(cfg['drill']['wrist_to_tip_pos'], dtype=float))
    # Frame placement is relative to the parent joint, not the parent frame.
    tip_frame_name = 'mapping_drill_tip'
    drill_id = model.addFrame(pin.Frame(
        tip_frame_name, wrist_frame.parentJoint, wrist_id,
        wrist_frame.placement * tool_transform, pin.FrameType.OP_FRAME))
    configurations = [Configuration(model, model.createData(), pin.neutral(model)) for _ in range(n)]
    ik = cfg['ik']
    tasks = [[FrameTask(probe_frame, position_cost=ik['probe_position_cost'], orientation_cost=ik['probe_orientation_cost']),
              FrameTask(tip_frame_name, position_cost=ik['drill_position_cost'], orientation_cost=ik['drill_orientation_cost'])] for _ in range(n)]
    sensor_cfg = Path(PACKAGE_DIR) / 'lab/sensors/cfgs'
    us = read_yaml(sensor_cfg / 'us_cfg.yaml')
    # Only geometry is used. No CT loading, ultrasound rendering, policy or logging.
    slicer = USSlicer(us, [nib.load(patient_dir / 'combined_label_map.nii.gz').get_fdata()],
                     [], False, [str(patient_dir)], n, bounds.tolist(), initial[0].tolist(),
                     device, read_yaml(sensor_cfg / 'label_conversion.yaml'), us['image_size'], us['resolution'],
                     img_thickness=1, roll_adj=cfg['probe']['roll_adj'], visualize=False,
                     sim_mode='conv', height=rc['height'], height_img=rc['height_img'])
    pose_converter = GTDiscreteMotionGenerator(
        goal_cmd_pose=tensor(bounds.mean(axis=0)).repeat(n, 1),
        scale=tensor(cfg['probe']['motion_scale']), num_envs=n,
        surface_map_list=slicer.surface_map_list,
        surface_normal_list=slicer.surface_normal_list,
        label_res=pc['label_res'], US_height=slicer.height)
    viewer = VertebraViewer(n, 1, [str(patient_dir / f"{pc['target_anatomy']}.stl")],
                           [str(patient_dir / f"standard_right_traj_{pc['target_anatomy'][-2:]}.stl")],
                           if_vis=False, res=pc['label_res'], device=device)
    wrist_offset = tensor(cfg['drill']['wrist_to_tip_pos']).repeat(n, 1)
    wrist_rot = tensor(quaternion('YXZ', cfg['drill']['wrist_to_tip_euler_YXZ'])).repeat(n, 1)
    direction = viewer.traj_drct / torch.linalg.vector_norm(viewer.traj_drct, dim=-1, keepdim=True).clamp_min(1e-9)
    # Specify insertion direction independently of the initial robot posture.
    # A hemisphere choice based on tip0 would cancel a reversal of the tool frame.
    z = direction * cfg['drill']['trajectory_direction_sign']
    up = tensor([0, 1, 0]).repeat(n, 1)
    up = torch.where((up * z).sum(-1, keepdim=True).abs() > .95, tensor([1, 0, 0]).repeat(n, 1), up)
    x = torch.nn.functional.normalize(torch.cross(up, z, dim=-1), dim=-1)
    tip_q_h = quat_from_matrix(torch.stack([x, torch.cross(z, x, dim=-1), z], -1))
    # Independent seeded stream: changing drill noise does not change probe commands.
    drill_rng = np.random.default_rng(np.random.SeedSequence([sc['seed'], 1]))
    noise_m = cfg['drill'].get('tip_along_z_noise_m', 0.0)
    if noise_m < 0:
        raise ValueError('drill.tip_along_z_noise_m must be nonnegative.')
    tip_offsets = cfg['drill']['tip_along_z_m'] + drill_rng.uniform(-noise_m, noise_m, n)
    cfg['resolved_tip_offsets_m'] = tip_offsets.tolist()
    tip_p_h = viewer.human_to_traj_pos.clone() + tensor(tip_offsets)[:, None] * matrix_from_quat(tip_q_h)[:, :, 2]
    us_rotation = tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).expand(n, -1, -1)
    rng = np.random.default_rng(sc['seed'])
    command = rng.uniform(initial[0], initial[1], size=(n, 3))
    if fixed_angle is not None:
        command[:, 2] = fixed_angle
    velocity = np.zeros_like(command)
    hold = max(1, round(cfg['probe']['command_hold_seconds'] / dt))
    steps = math.ceil(sc['episode_seconds'] / dt)
    records = ([], [], [], [])
    diagnostics = {'commands': [], 'probe_pose_h': [], 'probe_pose_us_h': []}
    cfg['map_coordinates'] = 'GTDiscreteMotionGenerator.human_cmd_state_from_ee_pose on measured US pose in patient frame'
    threshold = np.array([cfg['mapping'][key] for key in ('probe_position_tolerance_m', 'probe_angle_tolerance_deg',
                                                         'drill_position_tolerance_m', 'drill_angle_tolerance_deg')])
    folder = Path(__file__).resolve().parent / 'results' / f"{pc['id']}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir(parents=True)
    limits = robot.data.joint_pos_limits[:, controlled]
    default = robot.data.default_joint_pos.clone()

    def errors(pose, target_p, target_q):
        pos = torch.linalg.vector_norm(pose[:, :3] - target_p, dim=-1)
        dot = (pose[:, 3:7] * target_q).sum(-1).abs().clamp(0, 1)
        return cpu(pos), cpu(2 * torch.acos(dot) * (180 / math.pi))

    for step in range(sc['pre_steps'] + steps):
        if not app.is_running():
            break
        t = step - sc['pre_steps']
        if t >= 0:
            if t % hold == 0:
                velocity = rng.uniform(-1, 1, (n, 3)) * np.asarray(cfg['probe']['velocity_max'])
            proposed = command + velocity * dt
            velocity = np.where((proposed < bounds[0]) | (proposed > bounds[1]), -velocity, velocity)
            command = reflect(proposed, bounds[0], bounds[1])
        if fixed_angle is not None:
            command[:, 2] = fixed_angle
            velocity[:, 2] = 0.
        h = human.data.root_state_w[:, :7]
        b = robot.data.root_link_state_w[:, :7]
        slicer.current_x_z_x_angle_cmd = tensor(command)
        slicer.roll_adj = torch.full((n, 1), cfg['probe']['roll_adj'], device=device)
        slicer.compute_world_ee_pose_from_cmd(h[:, :3], h[:, 3:7])
        probe_p, probe_q = combine(h[:, :3], h[:, 3:7], slicer.human_to_ee_target_pos, slicer.human_to_ee_target_quat)
        probe_q = quat_from_matrix(matrix_from_quat(probe_q) @ us_rotation.transpose(1, 2))
        tip_p, tip_q = combine(h[:, :3], h[:, 3:7], tip_p_h, tip_q_h)
        poses = [relative(b[:, :3], b[:, 3:7], p, q) for p, q in [(probe_p, probe_q), (tip_p, tip_q)]]
        poses = [(cpu(p), cpu(matrix_from_quat(q))) for p, q in poses]
        current = cpu(robot.data.joint_pos[:, controlled])
        targets = np.empty_like(current)
        for env, configuration in enumerate(configurations):
            q = pin.neutral(model)
            q[q_idx] = current[env]
            configuration.update(q)
            for task, (p, rot) in zip(tasks[env], poses):
                task.set_target(pin.SE3(rot[env].astype(float), p[env].astype(float)))
            vel = solve_ik(configuration, tasks[env], dt, solver=ik['solver'], damping=ik['damping'], safety_break=False)
            configuration.integrate_inplace(vel, dt)
            targets[env] = configuration.q[q_idx]
        q_target = default.clone()
        q_target[:, controlled] = torch.clamp(tensor(targets), limits[..., 0] + ik['joint_margin'], limits[..., 1] - ik['joint_margin'])
        robot.set_joint_position_target(q_target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        if t < 0:
            continue
        probe_actual = robot.data.body_state_w[:, entities[0].body_ids[-1], :7]
        # Re-read the patient transform and actual probe pose AFTER physics.
        h_now = human.data.root_state_w[:, :7]
        actual_p_h, actual_q_h = relative(h_now[:, :3], h_now[:, 3:7],
                                         probe_actual[:, :3], probe_actual[:, 3:7])
        actual_us_rotation = matrix_from_quat(actual_q_h) @ us_rotation
        actual_us_q_h = quat_from_matrix(actual_us_rotation)
        # Exact conversion used for cur_cmd_pose in robotic_US_guidance_Pink.py.
        measured_pose = cpu(pose_converter.human_cmd_state_from_ee_pose(
            actual_p_h, actual_us_q_h)).copy()
        drill_actual = robot.data.body_state_w[:, entities[1].body_ids[-1], :7]
        actual_tip_p, actual_tip_q = combine(drill_actual[:, :3], drill_actual[:, 3:7], wrist_offset, wrist_rot)
        err = np.stack((*errors(probe_actual, probe_p, probe_q),
                        *errors(torch.cat([actual_tip_p, actual_tip_q], -1), tip_p, tip_q)), axis=-1)
        measured = cpu(robot.data.joint_pos[:, controlled])
        values = np.empty(n)
        for env, configuration in enumerate(configurations):
            q = pin.neutral(model)
            q[q_idx] = measured[env]
            jac = pin.computeFrameJacobian(model, configuration.data, q, probe_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
            drill_jac = None
            if cfg['mapping']['metric'] == 'drill_constrained':
                drill_jac = pin.computeFrameJacobian(model, configuration.data, q, drill_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
            values[env] = manipulability(jac, drill_jac)
        records[0].append(measured_pose)
        records[1].append(values)
        records[2].append(err)
        map_dims = np.asarray(cfg['mapping']['bins']) > 1
        in_range = np.all((measured_pose[:, map_dims] >= bounds[0, map_dims]) &
                          (measured_pose[:, map_dims] <= bounds[1, map_dims]), axis=-1)
        drill_ok = np.all(err[:, 2:] <= threshold[2:], axis=-1)
        records[3].append(drill_ok & in_range & np.isfinite(values))
        diagnostics['commands'].append(command.copy())
        diagnostics['probe_pose_h'].append(cpu(torch.cat([actual_p_h, actual_q_h], -1)).copy())
        diagnostics['probe_pose_us_h'].append(cpu(torch.cat([actual_p_h, actual_us_q_h], -1)).copy())
    if records[0]:
        errors_all = np.asarray(records[2])
        medians = np.median(errors_all, axis=(0, 1)) * [1000, 1, 1000, 1]
        passing = np.mean(errors_all <= threshold, axis=(0, 1)) * 100
        print('Tracking medians [probe mm, probe deg, tip mm, tip deg]:', medians.round(3))
        print('Passing each threshold [%] (probe diagnostic only):', passing.round(1))
        save_results(folder, records, bounds, cfg, diagnostics)
    else:
        print('Stopped before exploration; no map samples collected.')


def main():
    from isaaclab.app import AppLauncher
    from ruamel.yaml import YAML
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_suffix('.yaml'))
    parser.add_argument('--num_envs', type=int)
    parser.add_argument('--episode_seconds', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--patient', type=str)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    with args.config.open() as f:
        cfg = YAML(typ='safe').load(f)
    for key in ('num_envs', 'episode_seconds', 'seed'):
        if getattr(args, key) is not None:
            cfg['sim'][key] = getattr(args, key)
    if args.patient:
        cfg['patient']['id'] = args.patient
    if cfg['sim']['num_envs'] < 2 or cfg['sim']['episode_seconds'] <= 0 or cfg['sim']['dt'] <= 0:
        parser.error('Use at least two environments and positive duration/dt.')
    if cfg['sim']['pre_steps'] < 1 or any(b < 2 for b in cfg['mapping']['bins'][:2]) or cfg['mapping']['bins'][2] < 1:
        parser.error('pre_steps must be positive; x/z need >=2 bins, angle >=1.')
    if cfg['mapping']['metric'] not in ('yoshikawa', 'drill_constrained'):
        parser.error('Unknown manipulability metric.')
    if cfg['drill'].get('trajectory_direction_sign') not in (-1, 1):
        parser.error('drill.trajectory_direction_sign must be +1 or -1; use the updated YAML.')
    launcher = AppLauncher(args)
    try:
        run(args, cfg, launcher.app)
    finally:
        launcher.app.close()


if __name__ == '__main__':
    main()
