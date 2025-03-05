
import numpy as np
import plotly.graph_objects as go

def visualize_batch_plotly(example, output_file="camera_visualization.html"):
    fig = go.Figure()

    # Extract data
    context_extrinsics = example["context"]["extrinsics"]  # Shape: [n_context, 4, 4]
    target_extrinsics = example["target"]["extrinsics"]  # Shape: [n_target, 4, 4]
    refinement_extrinsics = example["refinement"]["extrinsics"] if example["refinement"] else None  # Shape: [n_target, n_refinement, 4, 4]

    context_intrinsics = example["context"]["intrinsics"]  # Shape: [n_context, 3, 3]
    target_intrinsics = example["target"]["intrinsics"]  # Shape: [n_target, 3, 3]
    refinement_intrinsics = example["refinement"]["intrinsics"] if example["refinement"] else None  # Shape: [n_target, n_refinement, 3, 3] or [n_target, 3, 3]

    # Define colors
    context_color = 'blue'
    target_colors = ['red', 'green', 'purple', 'orange', 'cyan']
    refinement_colors = ['pink', 'lightgreen', 'lightblue', 'yellow', 'gray']

    # Helper function to add a camera frustum
    def add_camera(fig, K, RT, color):
        R = RT[:3, :3]
        t = RT[:3, 3]

        # Define frustum points in image space
        img_corners = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]).T
        K_inv = np.linalg.inv(K)
        cam_corners = K_inv @ img_corners
        cam_corners /= cam_corners[2]  # Normalize z
        world_corners = (R @ cam_corners) + t.reshape(3, 1)

        # Draw frustum edges
        for i in range(4):
            fig.add_trace(go.Scatter3d(
                x=[t[0], world_corners[0, i]],
                y=[t[1], world_corners[1, i]],
                z=[t[2], world_corners[2, i]],
                mode='lines',
                line=dict(color=color, width=3)
            ))

        # Draw camera center
        fig.add_trace(go.Scatter3d(
            x=[t[0]], y=[t[1]], z=[t[2]],
            mode='markers',
            marker=dict(size=6, color=color, symbol='diamond')
        ))

    # Add context cameras
    for i in range(len(context_extrinsics)):
        add_camera(fig, context_intrinsics[i], context_extrinsics[i], context_color)

    # Add target cameras
    for i in range(len(target_extrinsics)):
        add_camera(fig, target_intrinsics[i], target_extrinsics[i], target_colors[i % len(target_colors)])

        # Add refinement cameras for this target
        if refinement_extrinsics is not None:
            n_refinement = len(refinement_extrinsics[i])
            for j in range(n_refinement):
                # Ensure correct indexing of intrinsics
                if refinement_intrinsics.ndim == 4:  # Case: (n_target, n_refinement, 3, 3)
                    K_ref = refinement_intrinsics[i, j]
                else:  # Case: (n_target, 3, 3), use the same intrinsics for all refinements per target
                    K_ref = refinement_intrinsics[i]
                
                RT_ref = refinement_extrinsics[i, j]
                add_camera(fig, K_ref, RT_ref, refinement_colors[i % len(refinement_colors)])  # Same color per target

    # Configure 3D Layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(range=[-5, 5]), yaxis=dict(range=[-5, 5]), zaxis=dict(range=[-5, 5])
        ),
        title="Camera Poses and Frustums (Interactive)",
    )

    # Save as HTML
    fig.write_html(output_file)
    print(f"Saved interactive visualization to {output_file}")

# Example Usage
example = {
    "context": {
        "extrinsics": np.random.randn(3, 4, 4),  # 3 context cameras
        "intrinsics": np.repeat(np.eye(3)[None, :, :], 3, axis=0),  # Same intrinsics for all
    },
    "target": {
        "extrinsics": np.random.randn(2, 4, 4),  # 2 target cameras
        "intrinsics": np.repeat(np.eye(3)[None, :, :], 2, axis=0),
    },
    "refinement": {
        "extrinsics": np.random.randn(2, 3, 4, 4),  # 2 targets, each with 3 refinement views
        "intrinsics": np.repeat(np.eye(3)[None, None, :, :], 2, axis=0),  # Adjusted dimensions
    }
}

visualize_batch_plotly(example)
