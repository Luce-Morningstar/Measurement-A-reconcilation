import os
os.environ["CUPY_NVCC_GENERATE_CODE"] = "--std=c++17"
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')  # Suppress matplotlib warnings

import cupy as cp
import healpy as hp
from scipy.signal import correlate
from scipy.special import rel_entr
from cupyx.scipy.ndimage import convolve

from healpy.sphtfunc import map2alm, alm2cl
# Import the actual simulation modules
try:
    CUPY_AVAILABLE = True
    print("CuPy and HEALPix successfully imported")
except ImportError as e:
    print(f"Warning: CuPy/HEALPix not available, using NumPy fallback: {e}")
    import numpy as cp
    CUPY_AVAILABLE = False
    
    # Mock HEALPix functions for fallback
    class MockHealPy:
        @staticmethod
        def nside2npix(nside):
            return 12 * nside * nside
        @staticmethod
        def ang2pix(nside, theta, phi):
            return np.zeros(len(theta), dtype=int)
        @staticmethod
        def read_map(filename, **kwargs):
            return np.random.random(12 * 256 * 256)
        @staticmethod
        def ud_grade(map_in, nside_out):
            return np.random.random(12 * nside_out * nside_out)
        @staticmethod
        def anafast(map_in, **kwargs):
            return np.random.random(500)
    
    hp = MockHealPy()
    
    def map2alm(map_in, **kwargs):
        return np.random.random(500) + 1j * np.random.random(500)
    
    def alm2cl(alm):
        return np.random.random(len(alm))
        
    try:
        from scipy.special import rel_entr
    except ImportError:
        def rel_entr(p, q):
            return p * np.log(p / q)

class LilithSimulation:
    """Core simulation class integrating Lilith 1.0 with real-time analysis"""
    def make_gravity_kernel(self):
        kernel = cp.zeros((3, 3, 3), dtype=cp.float32)
        center = cp.array([1, 1, 1])

        for x in range(3):
            for y in range(3):
                for z in range(3):
                    pos = cp.array([x, y, z])
                    dist = cp.linalg.norm(pos - center)
                    if dist > 0:
                        kernel[x, y, z] = 1.0 / (dist**2)
        kernel /= cp.sum(kernel)  # Normalize to prevent runaway force
        return kernel

    def __init__(self, params, output_queue, custom_output_dir=None):
        self.params = params
        self.output_queue = output_queue
        self.running = False
        self.step = 0
        self.custom_output_dir = custom_output_dir
        self.files_saved_count = 0
        self.gravity_kernel = self.make_gravity_kernel()

        # Create output directory
        self.setup_output_directory()
        
        # Initialize simulation state
        self.initialize_simulation()
        
        # Load Planck data for comparison
        self.load_planck_data()
    import cupy as cp


    def setup_output_directory(self):
        """Create output directory for saving results"""
        if self.custom_output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(self.custom_output_dir, f"lilith_run_{timestamp}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"lilith_gui_output_{timestamp}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created: {self.output_dir}")
        
    def initialize_simulation(self):
        """Initialize the simulation with current parameters"""
        size = self.params['size']
        max_layers = self.params['max_layers']
        n_obs = self.params['n_obs']
        
        # Initialize layers
        self.M_layers = []
        self.M_prev_layers = []
        self.M_i_layers = []
        self.rho_obs_layers = []
        self.shell_masks = []
        self.shell_surfaces = []
        self.radius_shells = []
        self.observer_states = []
        self.nucleation_fields = []
        self.memory_fields = []
        
        # Generate fractal layers
        for i in range(max_layers):
            scale = self.params['shell_scale_factor'] ** i
            center = size // 2
            xg, yg, zg = cp.meshgrid(cp.arange(size), cp.arange(size), cp.arange(size), indexing='ij')
            dx, dy, dz = xg - center, yg - center, zg - center
            radius_grid = cp.sqrt(dx**2 + dy**2 + dz**2)
            radius_shell = radius_grid.astype(cp.int32)
            shell_max = int(radius_grid.max() * scale)
            mask = (radius_grid <= shell_max).astype(cp.float32)
            surface = ((radius_grid >= shell_max - 1.5) & (radius_grid <= shell_max)).astype(cp.float32)

            M = self.white_noise_field((size, size, size)) * 0.1 * (1.0 / (1 + i))
            M_prev = M.copy()
            M_i = self.white_noise_field((size, size, size), scale=0.001)
            rho_obs = cp.zeros_like(M)

            # Initialize observers
            ob_x = cp.random.randint(0, size, n_obs)
            ob_y = cp.random.randint(0, size, n_obs)
            ob_z = cp.random.randint(0, size, n_obs)
            ob_age = cp.zeros(n_obs, dtype=cp.int32)
            ob_fn = cp.zeros(n_obs, dtype=cp.int32)
            ob_alive = cp.ones(n_obs, dtype=cp.bool_)
            ob_mob = cp.ones(n_obs, dtype=cp.float32)

            self.M_layers.append(M * mask)
            self.M_prev_layers.append(M_prev * mask)
            self.M_i_layers.append(M_i * mask)
            self.rho_obs_layers.append(rho_obs)
            self.radius_shells.append(radius_shell)
            self.shell_masks.append(mask)
            self.shell_surfaces.append(surface)
            self.observer_states.append({
                "x": ob_x, "y": ob_y, "z": ob_z, "age": ob_age, 
                "fn": ob_fn, "alive": ob_alive, "mobility": ob_mob
            })
            self.nucleation_fields.append(cp.zeros_like(M))
            self.memory_fields.append(cp.zeros_like(M))
            
        # Store grid coordinates for projections
        self.dx, self.dy, self.dz = dx, dy, dz
        
    def white_noise_field(self, shape, scale=0.1):
        """Generate white noise field"""
        noise = cp.random.normal(loc=0.0, scale=scale, size=shape)
        freq_noise = cp.fft.fftn(noise)
        random_phase = cp.exp(2j * cp.pi * cp.random.rand(*shape))
        filtered = cp.real(cp.fft.ifftn(freq_noise * random_phase))
        return filtered
        
    def laplacian_3d(self, F):
        """3D Laplacian operator"""
        return (
            cp.roll(F, 1, axis=0) + cp.roll(F, -1, axis=0) +
            cp.roll(F, 1, axis=1) + cp.roll(F, -1, axis=1) +
            cp.roll(F, 1, axis=2) + cp.roll(F, -1, axis=2) -
            6 * F
        )
        
    def observer_drift(self, M, ob, radius_shell, shell_max):
        """Observer movement and dynamics"""
        pot = M + 0.5 * self.laplacian_3d(M)
        grad_x, grad_y, grad_z = cp.gradient(pot)
        gx = grad_x[ob["x"], ob["y"], ob["z"]]
        gy = grad_y[ob["x"], ob["y"], ob["z"]]
        gz = grad_z[ob["x"], ob["y"], ob["z"]]
        norm = cp.sqrt(gx**2 + gy**2 + gz**2) + 1e-6

        ob["mobility"] *= self.params['observer_mobility_decay']

        # Cohesion behavior
        x_c, y_c, z_c = ob["x"], ob["y"], ob["z"]
        x_mean, y_mean, z_mean = cp.mean(x_c), cp.mean(y_c), cp.mean(z_c)
        cx = x_mean - x_c
        cy = y_mean - y_c
        cz = z_mean - z_c
        c_norm = cp.sqrt(cx**2 + cy**2 + cz**2) + 1e-6
        cohesion_weight = 0.9
        gx = (1 - cohesion_weight) * gx + cohesion_weight * (cx / c_norm)
        gy = (1 - cohesion_weight) * gy + cohesion_weight * (cy / c_norm)
        gz = (1 - cohesion_weight) * gz + cohesion_weight * (cz / c_norm)

        norm = cp.sqrt(gx**2 + gy**2 + gz**2) + 1e-6
        step_size = self.params.get('step_size', 0.5)
        size = self.params['size']
        
        x_new = cp.clip(ob["x"] + ob["mobility"] * step_size * (gx / norm), 0, size - 1).astype(cp.int32)
        y_new = cp.clip(ob["y"] + ob["mobility"] * step_size * (gy / norm), 0, size - 1).astype(cp.int32)
        z_new = cp.clip(ob["z"] + ob["mobility"] * step_size * (gz / norm), 0, size - 1).astype(cp.int32)

        # Handle shell boundaries
        r_obs = radius_shell[x_new, y_new, z_new]
        shell_hit = (r_obs >= shell_max)
        x_new[shell_hit] = size // 2
        y_new[shell_hit] = size // 2
        z_new[shell_hit] = size // 2

        return x_new, y_new, z_new
        
    def load_planck_data(self):
        """Load Planck CMB data for comparison"""
        try:
            # Try to load local Planck data - check multiple possible filenames
            planck_fits_files = ["SMICA_CMB.FITS", "smica_cmb.fits", "COM_CMB_IQU-smica_1024_R2.02_full.fits"]
            planck_cl_files = ["COM_PowerSpect_CMB-TT-full_R3.01.txt", "planck_2018_cls.txt"]
            
            self.planck_map = None
            self.planck_cl = None
            
            # Try to load FITS file
            for fname in planck_fits_files:
                if os.path.exists(fname):
                    try:
                        print(f"Loading Planck map from {fname}")
                        self.planck_map = hp.read_map(fname, field=0, verbose=False)
                        self.planck_map = hp.ud_grade(self.planck_map, nside_out=self.params['nside'])
                        print(f"Successfully loaded Planck map with nside={self.params['nside']}")
                        break
                    except Exception as e:
                        print(f"Failed to load {fname}: {e}")
                        continue
                        
            # Try to load power spectrum file
            for fname in planck_cl_files:
                if os.path.exists(fname):
                    try:
                        print(f"Loading Planck power spectrum from {fname}")
                        data = np.loadtxt(fname)
                        self.planck_cl = data[:, 1] if data.shape[1] > 1 else data
                        print(f"Successfully loaded Planck Cl with {len(self.planck_cl)} multipoles")
                        break
                    except Exception as e:
                        print(f"Failed to load {fname}: {e}")
                        continue
                        
            # Generate Planck Cl from map if we have map but no Cl file
            if self.planck_map is not None and self.planck_cl is None:
                try:
                    print("Generating power spectrum from Planck map...")
                    self.planck_cl = hp.anafast(self.planck_map, lmax=min(512, 3*self.params['nside']-1))
                    print(f"Generated Planck Cl with {len(self.planck_cl)} multipoles")
                except Exception as e:
                    print(f"Failed to generate Cl from map: {e}")
                    
        except Exception as e:
            print(f"Warning: Could not load Planck data: {e}")
            self.planck_map = None
            self.planck_cl = None
            
    def compute_metrics(self):
        """Compute real-time analysis metrics"""
        metrics = {}
        
        # Combine shell data for projection
        size = self.params['size']
        combined_shell = cp.zeros((size, size, size))
        for i in range(len(self.M_layers)):
            combined_shell += self.M_layers[i] * self.shell_surfaces[i]
            
        # Convert to HEALPix projection
        shell_energy = float(cp.sum(combined_shell))
        metrics['shell_energy'] = shell_energy
        
        if shell_energy > 1e-6:
            # Create HEALPix projection
            r_grid = cp.sqrt(self.dx**2 + self.dy**2 + self.dz**2) + 1e-6
            valid_mask = combined_shell > 0
            
            if cp.sum(valid_mask) > 0:
                dz_valid = self.dz[valid_mask]
                dy_valid = self.dy[valid_mask]
                dx_valid = self.dx[valid_mask]
                r_valid = r_grid[valid_mask]
                theta = cp.arccos(dz_valid / r_valid)
                phi = cp.arctan2(dy_valid, dx_valid) % (2 * cp.pi)
                weights = combined_shell[valid_mask]

                # Convert to numpy for HEALPix
                theta_np = cp.asnumpy(theta)
                phi_np = cp.asnumpy(phi)
                weights_np = cp.asnumpy(weights)

                npix = hp.nside2npix(self.params['nside'])
                pix = hp.ang2pix(self.params['nside'], theta_np, phi_np)
                proj = np.bincount(pix, weights=weights_np, minlength=npix)
                
                # Compute power spectrum
                if np.std(proj) > 1e-6:
                    try:
                        alm = map2alm(proj, lmax=min(256, self.params['nside']))
                        cl = alm2cl(alm)
                        
                        # Normalize for entropy calculation
                        # Normalize for entropy calculation
                        cl_norm = cl / (np.sum(cl) + 1e-12) 

                        # Compare with Planck if available
                        if self.planck_cl is not None and len(cl) > 10:
                            planck_truncated = self.planck_cl[:len(cl)] / 1e3
                            planck_norm = planck_truncated / (np.sum(planck_truncated) + 1e-12)

                            # Clip both distributions to avoid log(0) fuckery
                            eps = 1e-12
                            cl_norm = np.clip(cl_norm, eps, 1.0)
                            planck_norm = np.clip(planck_norm, eps, 1.0)

                            # KL divergence
                            kl_div = np.sum(rel_entr(cl_norm, planck_norm))
                            metrics['kl_divergence'] = float(kl_div) if not np.isnan(kl_div) else 0.0

                            # Correlation
                            corr = np.corrcoef(cl, planck_truncated)[0, 1]
                            metrics['correlation'] = float(corr) if not np.isnan(corr) else 0.0
                        else:
                            metrics['kl_divergence'] = 0.0
                            metrics['correlation'] = 0.0

                            
                        # Entropy (now cl_norm is always defined)
                        entropy = -np.sum(cl_norm * np.log(cl_norm + 1e-12))
                        metrics['entropy'] = float(entropy)
                        
                        # Store projection for visualization
                        self.current_projection = proj
                        self.current_cl = cl
                        
                        # Save data every 10 steps
                        if self.step % 10 == 0:
                            self.save_projection_data(proj)
                            
                        # Save power spectrum comparison every 50 steps
                        if self.step % 50 == 0:
                            self.save_power_spectrum_comparison(cl)
                        
                    except Exception as e:
                        print(f"Error computing power spectrum at step {self.step}: {e}")
                        metrics['kl_divergence'] = 0.0
                        metrics['correlation'] = 0.0
                        metrics['entropy'] = 0.0
                else:
                    metrics['kl_divergence'] = 0.0
                    metrics['correlation'] = 0.0
                    metrics['entropy'] = 0.0
            else:
                metrics['kl_divergence'] = 0.0
                metrics['correlation'] = 0.0
                metrics['entropy'] = 0.0
        else:
            metrics['kl_divergence'] = 0.0
            metrics['correlation'] = 0.0
            metrics['entropy'] = 0.0
            
        # Observer metrics
        total_observers = sum(len(obs["x"]) for obs in self.observer_states)
        metrics['observer_count'] = total_observers
        
        # Field metrics
        field_variance = float(cp.var(self.M_layers[0]))
        metrics['field_variance'] = field_variance
        
        # Coherence index
        if len(self.M_layers) > 0:
            coherence = float(cp.mean(cp.abs(self.M_layers[0] - self.M_prev_layers[0])))
            metrics['coherence_index'] = coherence
        else:
            metrics['coherence_index'] = 0.0
            
        return metrics
        
    def save_projection_data(self, projection):
        """Save projection data and create Mollweide plot"""
        try:
            # Save NPY file
            npy_filename = os.path.join(self.output_dir, f"projection_{self.step:06d}.npy")
            np.save(npy_filename, projection)
            self.files_saved_count += 1
            
            # Create and save Mollweide plot if HEALPix is available
            if CUPY_AVAILABLE:
                plt.figure(figsize=(12, 6))
                try:
                    hp.mollview(np.log1p(np.abs(projection)), 
                               title=f"Lilith Field - Step {self.step}", 
                               cmap="inferno", cbar=True, hold=True)
                    plot_filename = os.path.join(self.output_dir, f"mollweide_{self.step:06d}.png")
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    self.files_saved_count += 1
                except Exception as e:
                    print(f"HEALPix mollview error at step {self.step}: {e}")
                    # Fallback to simple plot
                    self.save_simple_projection_plot(projection)
            else:
                self.save_simple_projection_plot(projection)
                
            # Send file count update to GUI
            self.output_queue.put(('files_saved', {'count': self.files_saved_count}))
                
        except Exception as e:
            print(f"Error saving projection data at step {self.step}: {e}")
            
    def save_simple_projection_plot(self, projection):
        """Save a simple 2D projection plot as fallback"""
        try:
            plt.figure(figsize=(10, 8))
            
            # Create 2D representation
            side_len = int(np.sqrt(len(projection) / 12))
            if side_len < 32:
                side_len = 64
            
            grid_size = min(128, side_len)
            if len(projection) >= grid_size * grid_size:
                data_2d = projection[:grid_size*grid_size].reshape((grid_size, grid_size))
            else:
                padded_data = np.zeros(grid_size * grid_size)
                padded_data[:len(projection)] = projection
                data_2d = padded_data.reshape((grid_size, grid_size))
            
            plt.imshow(np.log1p(np.abs(data_2d)), cmap='inferno', aspect='auto')
            plt.colorbar(label='Log(1 + Field Strength)')
            plt.title(f"Lilith Field Projection - Step {self.step}")
            plt.xlabel('Longitude (projected)')
            plt.ylabel('Latitude (projected)')
            
            plot_filename = os.path.join(self.output_dir, f"projection_2d_{self.step:06d}.png")
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            self.files_saved_count += 1
            
        except Exception as e:
            print(f"Error saving simple projection plot at step {self.step}: {e}")
            
    def save_power_spectrum_comparison(self, cl_data):
        """Save power spectrum comparison with Planck"""
        try:
            plt.figure(figsize=(12, 8))
            
            ell = np.arange(len(cl_data))
            plt.loglog(ell[1:], cl_data[1:], label='Lilith Simulation', color='red', linewidth=2)
            
            # Add Planck comparison if available
            if self.planck_cl is not None:
                planck_truncated = self.planck_cl[:len(cl_data)]
                plt.loglog(ell[1:len(planck_truncated)], planck_truncated[1:], 
                          label='Planck 2018 CMB', linestyle='--', color='blue', linewidth=2)
                
                # Calculate correlation for the plot
                if len(cl_data) > 10:
                    corr = np.corrcoef(cl_data, planck_truncated)[0, 1]
                    plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                            transform=plt.gca().transAxes, fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            
            plt.xlabel('Multipole moment $\ell$', fontsize=14)
            plt.ylabel('C_\u2113 [\u03BCK\u00b2]', fontsize=14)
            plt.title(f'Angular Power Spectrum Comparison - Step {self.step}', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            plot_filename = os.path.join(self.output_dir, f"power_spectrum_{self.step:06d}.png")
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            self.files_saved_count += 1
            
            # Also save the Cl data as NPY
            cl_filename = os.path.join(self.output_dir, f"power_spectrum_{self.step:06d}.npy")
            np.save(cl_filename, cl_data)
            self.files_saved_count += 1
            
            # Send file count update to GUI
            self.output_queue.put(('files_saved', {'count': self.files_saved_count}))
            
        except Exception as e:
            print(f"Error saving power spectrum comparison at step {self.step}: {e}")
            
    def save_final_state(self):
        """Save final simulation state"""
        try:
            # Save final field data
            for i, M_layer in enumerate(self.M_layers):
                field_filename = os.path.join(self.output_dir, f"final_field_layer_{i}.npy")
                if hasattr(M_layer, 'get'):  # CuPy array
                    np.save(field_filename, M_layer.get())
                else:  # NumPy array
                    np.save(field_filename, M_layer)
                    
            # Save observer states
            for i, observer_state in enumerate(self.observer_states):
                obs_filename = os.path.join(self.output_dir, f"final_observers_layer_{i}.npy")
                obs_data = {}
                for key, value in observer_state.items():
                    if hasattr(value, 'get'):  # CuPy array
                        obs_data[key] = value.get()
                    else:  # NumPy array
                        obs_data[key] = value
                np.save(obs_filename, obs_data)
                
            # Save simulation parameters
            params_filename = os.path.join(self.output_dir, "simulation_parameters.json")
            with open(params_filename, 'w') as f:
                json.dump(self.params, f, indent=2)
                
            print(f"Final simulation state saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error saving final state: {e}")
        
    def simulation_step(self):
        """Execute one simulation step"""
        try:
            size = self.params['size']
            delta_t = self.params['delta_t']
            c = self.params['c']
            D = self.params['D']
            lam = self.params['lam']
            kappa = self.params['kappa']
            
            for i in range(len(self.M_layers)):
                M, M_prev, M_i, rho_obs = (self.M_layers[i], self.M_prev_layers[i], 
                                           self.M_i_layers[i], self.rho_obs_layers[i])
                ob = self.observer_states[i]
                radius_shell = self.radius_shells[i]
                shell_max = int(radius_shell.max())

                # Observer dynamics
                try:
                    ob_x, ob_y, ob_z = self.observer_drift(M, ob, radius_shell, shell_max)
                    ob["x"], ob["y"], ob["z"] = ob_x, ob_y, ob_z
                except Exception as e:
                    print(f"Observer drift error at step {self.step}: {e}")
                    # Skip observer updates but continue with field evolution

                # Observer replication in coherent zones
                try:
                    coherence_zone = cp.abs(M - M_prev) < 0.01
                    coherent_indices = cp.where(coherence_zone)
                    if len(coherent_indices[0]) > 10:
                        n_new = min(5, len(coherent_indices[0]))
                        if n_new > 0:
                            sampled = cp.random.choice(len(coherent_indices[0]), size=n_new, replace=False)
                            new_x = coherent_indices[0][sampled]
                            new_y = coherent_indices[1][sampled]
                            new_z = coherent_indices[2][sampled]
                            ob["x"] = cp.concatenate((ob["x"], new_x))
                            ob["y"] = cp.concatenate((ob["y"], new_y))
                            ob["z"] = cp.concatenate((ob["z"], new_z))
                            ob["age"] = cp.concatenate((ob["age"], cp.zeros(len(new_x), dtype=cp.int32)))
                            ob["fn"] = cp.concatenate((ob["fn"], cp.zeros(len(new_x), dtype=cp.int32)))
                            ob["alive"] = cp.concatenate((ob["alive"], cp.ones(len(new_x), dtype=cp.bool_)))
                            ob["mobility"] = cp.concatenate((ob["mobility"], cp.ones(len(new_x), dtype=cp.float32)))
                except Exception as e:
                    print(f"Observer replication error at step {self.step}: {e}")

                # Update observer density
                try:
                    rho_obs *= 0.1
                    if len(ob["x"]) > 0:
                        # Ensure indices are valid
                        valid_x = cp.clip(ob["x"], 0, size - 1)
                        valid_y = cp.clip(ob["y"], 0, size - 1)
                        valid_z = cp.clip(ob["z"], 0, size - 1)
                        rho_obs[valid_x, valid_y, valid_z] += 5 * cp.exp(-0.05 * self.step)
                except Exception as e:
                    print(f"Observer density update error at step {self.step}: {e}")

                # Field evolution
                try:
                    lap = self.laplacian_3d(M)
                    decay = -lam * M * float(min(self.step / 5.0, 1.0))
                    source = kappa * rho_obs
                    accel = c**2 * D * lap + decay + source

                    M_next = 2 * M - M_prev + delta_t**2 * accel
                    # === GRAVITY CLUMPING ===
                    try:
                        gravity_force = convolve(M_next, self.gravity_kernel, mode='reflect')
                        G_strength = 0.005  # you can tweak this like a freak
                        M_next += G_strength * gravity_force
                    except Exception as e:
                        print(f"Gravity clumping error at step {self.step}: {e}")

                    # Update nucleation fields
                    coherence = cp.abs(M - M_prev)
                    self.nucleation_fields[i] = cp.where((M > 0.05) & (coherence < 0.01), M, 0)
                    self.M_layers[i] = cp.clip(self.M_layers[i], 0.0, 1e3)

                    # Update layers
                    self.M_prev_layers[i] = M
                    self.M_layers[i] = M_next
                    self.M_i_layers[i] = M_i + 0.1 * self.laplacian_3d(M_i) - 0.01 * M_i
                except Exception as e:
                    print(f"Field evolution error at step {self.step}: {e}")
                    # If field evolution fails, try to continue with next layer

            self.step += 1
            
        except Exception as e:
            print(f"Critical simulation error at step {self.step}: {e}")
            raise  # Re-raise to stop simulation
        
    def run_simulation(self):
        """Main simulation loop"""
        self.running = True
        start_time = time.time()
        
        while self.running and self.step < self.params['steps']:
            try:
                self.simulation_step()
                
                # Compute metrics every 10 steps
                if self.step % 10 == 0:
                    metrics = self.compute_metrics()
                    metrics['step'] = self.step
                    metrics['elapsed_time'] = time.time() - start_time
                    self.output_queue.put(('metrics', metrics))
                    
                # Send visualization data every 50 steps
                if self.step % 50 == 0 and hasattr(self, 'current_projection'):
                    vis_data = {
                        'projection': self.current_projection,
                        'power_spectrum': getattr(self, 'current_cl', None),
                        'step': self.step
                    }
                    self.output_queue.put(('visualization', vis_data))
                    
                # Small delay to prevent GUI freezing
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Simulation error at step {self.step}: {e}")
                break
                
        # Save final state when simulation completes
        self.save_final_state()
        self.output_queue.put(('simulation_complete', {'final_step': self.step, 'output_dir': self.output_dir}))
        
    def stop(self):
        """Stop the simulation"""
        self.running = False


class LilithGUI:
    """Main GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Lilith 1.0 - Observer Field Dynamics")
        self.root.geometry("1400x900")
        
        # Initialize parameters
        self.init_parameters()
        
        # Threading
        self.simulation_thread = None
        self.simulation = None
        self.output_queue = queue.Queue()
        
        # GUI state
        self.running = False
        self.auto_randomize = tk.BooleanVar()
        self.randomize_interval = tk.IntVar(value=5000)
        self.custom_output_dir = None
        
        # Metrics storage
        self.metrics_history = []
        
        # Create GUI
        self.create_widgets()
        
        # Update initial status bar
        self.update_randomization_status()
        
        # Start update loop
        self.update_gui()
        
    def init_parameters(self):
        """Initialize simulation parameters with randomization ranges"""
        self.parameters = {
            'size': {'value': 128, 'min': 64, 'max': 256, 'step': 32, 'randomize': False, 'rand_min': 64, 'rand_max': 256},
            'steps': {'value': 10000, 'min': 1000, 'max': 50000, 'step': 1000, 'randomize': False, 'rand_min': 5000, 'rand_max': 25000},
            'delta_t': {'value': 0.349, 'min': 0.1, 'max': 1.0, 'step': 0.01, 'randomize': True, 'rand_min': 0.2, 'rand_max': 0.8},
            'c': {'value': 1.0, 'min': 0.5, 'max': 2.0, 'step': 0.1, 'randomize': False, 'rand_min': 0.8, 'rand_max': 1.5},
            'D': {'value': 0.25, 'min': 0.1, 'max': 1.0, 'step': 0.01, 'randomize': True, 'rand_min': 0.15, 'rand_max': 0.6},
            'lam': {'value': 8.5, 'min': 1.0, 'max': 20.0, 'step': 0.1, 'randomize': True, 'rand_min': 5.0, 'rand_max': 15.0},
            'kappa': {'value': 5.0, 'min': 0.0, 'max': 20.0, 'step': 0.1, 'randomize': True, 'rand_min': 2.0, 'rand_max': 12.0},
            'nside': {'value': 256, 'min': 128, 'max': 512, 'step': 128, 'randomize': False, 'rand_min': 128, 'rand_max': 512},
            'n_obs': {'value': 32, 'min': 8, 'max': 128, 'step': 8, 'randomize': True, 'rand_min': 16, 'rand_max': 64},
            'max_layers': {'value': 2, 'min': 1, 'max': 4, 'step': 1, 'randomize': False, 'rand_min': 2, 'rand_max': 3},
            'observer_lifetime': {'value': 400, 'min': 100, 'max': 1000, 'step': 50, 'randomize': True, 'rand_min': 200, 'rand_max': 800},
            'observer_decay_rate': {'value': 0.85, 'min': 0.1, 'max': 0.99, 'step': 0.01, 'randomize': True, 'rand_min': 0.7, 'rand_max': 0.95},
            'observer_mobility_decay': {'value': 0.50, 'min': 0.1, 'max': 0.95, 'step': 0.01, 'randomize': True, 'rand_min': 0.3, 'rand_max': 0.8},
            'shell_scale_factor': {'value': 0.5, 'min': 0.1, 'max': 0.9, 'step': 0.1, 'randomize': False, 'rand_min': 0.3, 'rand_max': 0.7},
            'step_size': {'value': 0.5, 'min': 0.1, 'max': 2.0, 'step': 0.1, 'randomize': True, 'rand_min': 0.3, 'rand_max': 1.0}

            'domain_decomposition': {'value': 1.0, 'min': 0.0, 'max': 1.0, 'step': 1.0, 'randomize': False, 'rand_min': 0.0, 'rand_max': 1.0},  
            'cooling_phase_2_decay': {'value': 1.5, 'min': 1.0, 'max': 3.0, 'step': 0.1, 'randomize': True, 'rand_min': 1.2, 'rand_max': 2.0},
            'cooling_phase_3_decay': {'value': 2.0, 'min': 1.5, 'max': 5.0, 'step': 0.1, 'randomize': True, 'rand_min': 1.8, 'rand_max': 3.0},
            'max_observers_per_domain': {'value': 1000, 'min': 100, 'max': 5000, 'step': 100, 'randomize': False, 'rand_min': 500, 'rand_max': 2000}
        }
        
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control Panel
        self.create_control_panel(control_frame)
        
        # Visualization Panel
        self.create_visualization_panel(viz_frame)
        
        # Status Bar  
        self.create_status_bar()
        
    def create_control_panel(self, parent):
        """Create the control panel"""
        # Title
        title_label = ttk.Label(parent, text="Lilith 1.0 Control", font=("Arial", 14, "bold"))
        title_label.pack(pady=5)
        
        # Main controls
        control_group = ttk.LabelFrame(parent, text="Simulation Control")
        control_group.pack(fill=tk.X, pady=5)
        
        button_frame = ttk.Frame(control_group)
        button_frame.pack(pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=2)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=2)
        
        # Status
        self.status_label = ttk.Label(control_group, text="Status: Ready")
        self.status_label.pack(pady=2)
        
        self.step_label = ttk.Label(control_group, text="Step: 0")
        self.step_label.pack(pady=2)
        
        # Randomization controls
        random_group = ttk.LabelFrame(parent, text="Parameter Randomization")
        random_group.pack(fill=tk.X, pady=5)
        
        button_frame = ttk.Frame(random_group)
        button_frame.pack(pady=2, fill=tk.X)
        
        randomize_button = ttk.Button(button_frame, text="Randomize Selected", command=self.randomize_parameters)
        randomize_button.pack(side=tk.LEFT, padx=2)
        
        toggle_all_button = ttk.Button(button_frame, text="Toggle All", command=self.toggle_all_randomization)
        toggle_all_button.pack(side=tk.LEFT, padx=2)
        
        save_profile_button = ttk.Button(button_frame, text="Save Profile", command=self.save_randomization_profile)
        save_profile_button.pack(side=tk.LEFT, padx=2)
        
        load_profile_button = ttk.Button(button_frame, text="Load Profile", command=self.load_randomization_profile)
        load_profile_button.pack(side=tk.LEFT, padx=2)
        
        # Output directory controls
        output_frame = ttk.Frame(random_group)
        output_frame.pack(pady=2, fill=tk.X)
        
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar(value="Auto-generated")
        self.output_dir_label = ttk.Label(output_frame, textvariable=self.output_dir_var, 
                                         font=("TkDefaultFont", 8), foreground="blue")
        self.output_dir_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        choose_dir_button = ttk.Button(output_frame, text="Choose", command=self.choose_output_directory)
        choose_dir_button.pack(side=tk.RIGHT)
        
        open_dir_button = ttk.Button(output_frame, text="Open", command=self.open_output_directory)
        open_dir_button.pack(side=tk.RIGHT, padx=(0, 5))
        
        auto_frame = ttk.Frame(random_group)
        auto_frame.pack(pady=2)
        
        auto_check = ttk.Checkbutton(auto_frame, text="Auto-randomize", variable=self.auto_randomize)
        auto_check.pack(side=tk.LEFT)
        
        ttk.Label(auto_frame, text="Interval (ms):").pack(side=tk.LEFT, padx=(10, 2))
        interval_entry = ttk.Entry(auto_frame, textvariable=self.randomize_interval, width=8)
        interval_entry.pack(side=tk.LEFT)
        
        # Randomization status
        self.randomization_status = ttk.Label(random_group, text="")
        self.randomization_status.pack(pady=2)
        
        # Parameter controls
        param_group = ttk.LabelFrame(parent, text="Parameters")
        param_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create scrollable parameter frame
        canvas = tk.Canvas(param_group, height=350)
        scrollbar = ttk.Scrollbar(param_group, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Bind mousewheel to canvas for better scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Parameter widgets
        self.param_widgets = {}
        for param_name, param_info in self.parameters.items():
            self.create_parameter_widget(scrollable_frame, param_name, param_info)
            
        # Update randomization status after all widgets are created
        self.update_randomization_status()
            
        # Metrics display
        metrics_group = ttk.LabelFrame(parent, text="Real-time Metrics")
        metrics_group.pack(fill=tk.X, pady=5)
        
        self.metrics_labels = {}
        metrics_names = ['KL Divergence', 'Correlation', 'Entropy', 'Observers', 'Shell Energy', 'Coherence']
        for name in metrics_names:
            label = ttk.Label(metrics_group, text=f"{name}: 0.000")
            label.pack(anchor=tk.W, padx=5)
            self.metrics_labels[name] = label
            
    def create_parameter_widget(self, parent, name, param_info):
        """Create a parameter control widget with randomization controls"""
        # Main frame with colored border for randomization status
        main_frame = ttk.Frame(parent, relief=tk.RIDGE, borderwidth=1)
        main_frame.pack(fill=tk.X, pady=2, padx=2)
        
        # Header frame for name and randomize toggle
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=2)
        
        # Parameter name and randomize checkbox
        name_frame = ttk.Frame(header_frame)
        name_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        randomize_var = tk.BooleanVar(value=param_info.get('randomize', False))
        randomize_check = ttk.Checkbutton(name_frame, text="", variable=randomize_var, 
                                         command=lambda: self.on_randomize_toggle(name, randomize_var.get()))
        randomize_check.pack(side=tk.LEFT)
        
        label = ttk.Label(name_frame, text=name.replace('_', ' ').title() + ":", 
                         font=("TkDefaultFont", 9, "bold" if param_info.get('randomize', False) else "normal"))
        label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Current value display
        value_label = ttk.Label(header_frame, text=f"{param_info['value']:.3f}", 
                               foreground="red" if param_info.get('randomize', False) else "black")
        value_label.pack(side=tk.RIGHT)
        
        # Main parameter control
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=2)
        
        # Current value scale
        var = tk.DoubleVar(value=param_info['value'])
        scale = ttk.Scale(control_frame, from_=param_info['min'], to=param_info['max'], 
                         variable=var, orient=tk.HORIZONTAL)
        scale.pack(fill=tk.X, pady=1)
        
        # Randomization range controls (initially hidden)
        range_frame = ttk.Frame(main_frame)
        if param_info.get('randomize', False):
            range_frame.pack(fill=tk.X, pady=2)
        
        # Range label
        range_label = ttk.Label(range_frame, text="Randomization Range:", font=("TkDefaultFont", 8))
        range_label.pack(anchor=tk.W)
        
        # Min range control
        min_range_frame = ttk.Frame(range_frame)
        min_range_frame.pack(fill=tk.X, pady=1)
        
        ttk.Label(min_range_frame, text="Min:", font=("TkDefaultFont", 8)).pack(side=tk.LEFT)
        min_var = tk.DoubleVar(value=param_info.get('rand_min', param_info['min']))
        min_scale = ttk.Scale(min_range_frame, from_=param_info['min'], to=param_info['max'], 
                             variable=min_var, orient=tk.HORIZONTAL)
        min_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        min_value_label = ttk.Label(min_range_frame, text=f"{min_var.get():.3f}", font=("TkDefaultFont", 8))
        min_value_label.pack(side=tk.RIGHT)
        
        # Max range control  
        max_range_frame = ttk.Frame(range_frame)
        max_range_frame.pack(fill=tk.X, pady=1)
        
        ttk.Label(max_range_frame, text="Max:", font=("TkDefaultFont", 8)).pack(side=tk.LEFT)
        max_var = tk.DoubleVar(value=param_info.get('rand_max', param_info['max']))
        max_scale = ttk.Scale(max_range_frame, from_=param_info['min'], to=param_info['max'], 
                             variable=max_var, orient=tk.HORIZONTAL)
        max_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        max_value_label = ttk.Label(max_range_frame, text=f"{max_var.get():.3f}", font=("TkDefaultFont", 8))
        max_value_label.pack(side=tk.RIGHT)
        
        # Update callbacks
        def update_value(*args):
            value = var.get()
            # Snap to step
            snapped = round(value / param_info['step']) * param_info['step']
            snapped = max(param_info['min'], min(param_info['max'], snapped))
            var.set(snapped)
            value_label.config(text=f"{snapped:.3f}")
            param_info['value'] = snapped
            
        def update_min_range(*args):
            value = min_var.get()
            snapped = round(value / param_info['step']) * param_info['step']
            snapped = max(param_info['min'], min(max_var.get(), snapped))
            min_var.set(snapped)
            min_value_label.config(text=f"{snapped:.3f}")
            param_info['rand_min'] = snapped
            
        def update_max_range(*args):
            value = max_var.get()
            snapped = round(value / param_info['step']) * param_info['step']
            snapped = min(param_info['max'], max(min_var.get(), snapped))
            max_var.set(snapped)
            max_value_label.config(text=f"{snapped:.3f}")
            param_info['rand_max'] = snapped
            
        var.trace('w', update_value)
        min_var.trace('w', update_min_range)
        max_var.trace('w', update_max_range)
        
        # Store widget references
        widget_data = {
            'var': var, 
            'label': value_label, 
            'info': param_info,
            'randomize_var': randomize_var,
            'randomize_check': randomize_check,
            'name_label': label,
            'range_frame': range_frame,
            'min_var': min_var,
            'max_var': max_var,
            'min_label': min_value_label,
            'max_label': max_value_label,
            'main_frame': main_frame
        }
        
        self.param_widgets[name] = widget_data
        
        # Update visual state
        self.update_parameter_visual_state(name)
        
    def create_visualization_panel(self, parent):
        """Create the visualization panel"""
        # Create notebook for different plots
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Mollweide projection tab
        moll_frame = ttk.Frame(notebook)
        notebook.add(moll_frame, text="Field Projection")
        
        self.moll_fig = Figure(figsize=(8, 4), dpi=100)
        self.moll_canvas = FigureCanvasTkAgg(self.moll_fig, moll_frame)
        self.moll_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Power spectrum tab
        ps_frame = ttk.Frame(notebook)
        notebook.add(ps_frame, text="Power Spectrum")
        
        self.ps_fig = Figure(figsize=(8, 6), dpi=100)
        self.ps_canvas = FigureCanvasTkAgg(self.ps_fig, ps_frame)
        self.ps_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Metrics history tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="Metrics History")
        
        self.metrics_fig = Figure(figsize=(8, 6), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, metrics_frame)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_status_bar(self):
        """Create the status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Left status info
        left_status = ttk.Frame(status_frame)
        left_status.pack(side=tk.LEFT)
        
        self.sim_id_label = ttk.Label(left_status, text="Simulation ID: None", font=("TkDefaultFont", 8))
        self.sim_id_label.pack(side=tk.LEFT, padx=5)
        
        self.field_res_label = ttk.Label(left_status, text="", font=("TkDefaultFont", 8))
        self.field_res_label.pack(side=tk.LEFT, padx=5)
        
        # Right status info  
        right_status = ttk.Frame(status_frame)
        right_status.pack(side=tk.RIGHT)
        
        self.randomize_count_label = ttk.Label(right_status, text="", font=("TkDefaultFont", 8), foreground="red")
        self.randomize_count_label.pack(side=tk.RIGHT, padx=5)
        
        self.sim_status_label = ttk.Label(right_status, text="READY", font=("TkDefaultFont", 8, "bold"), foreground="blue")
        self.sim_status_label.pack(side=tk.RIGHT, padx=5)
        
        self.files_saved_label = ttk.Label(right_status, text="Files saved: 0", font=("TkDefaultFont", 8), foreground="gray")
        self.files_saved_label.pack(side=tk.RIGHT, padx=5)
        
    def on_randomize_toggle(self, param_name, enabled):
        """Handle randomization toggle for a parameter"""
        param_info = self.parameters[param_name]
        param_info['randomize'] = enabled
        self.update_parameter_visual_state(param_name)
        self.update_randomization_status()
        
    def update_parameter_visual_state(self, param_name):
        """Update visual state of parameter widget based on randomization status"""
        widget_data = self.param_widgets[param_name]
        param_info = widget_data['info']
        is_randomized = param_info.get('randomize', False)
        
        # Update frame border color
        if is_randomized:
            widget_data['main_frame'].config(relief=tk.RIDGE, borderwidth=2)
            widget_data['name_label'].config(font=("TkDefaultFont", 9, "bold"), foreground="red")
            widget_data['label'].config(foreground="red")
            widget_data['range_frame'].pack(fill=tk.X, pady=2)
        else:
            widget_data['main_frame'].config(relief=tk.RIDGE, borderwidth=1)
            widget_data['name_label'].config(font=("TkDefaultFont", 9, "normal"), foreground="black")
            widget_data['label'].config(foreground="black")
            widget_data['range_frame'].pack_forget()
            
    def update_randomization_status(self):
        """Update the randomization status display"""
        enabled_params = [name for name, info in self.parameters.items() if info.get('randomize', False)]
        count = len(enabled_params)
        
        if count == 0:
            status_text = "No parameters set for randomization"
            color = "gray"
            status_bar_text = "Randomizing: 0 params"
        else:
            status_text = f"{count} parameters enabled: {', '.join(enabled_params[:3])}"
            if count > 3:
                status_text += f" (+{count-3} more)"
            color = "red"
            status_bar_text = f"Randomizing: {count} params ({', '.join(enabled_params[:2])}{'...' if count > 2 else ''})"
            
        self.randomization_status.config(text=status_text, foreground=color)
        
        # Update status bar if it exists
        if hasattr(self, 'randomize_count_label'):
            self.randomize_count_label.config(text=status_bar_text)
        
    def toggle_all_randomization(self):
        """Toggle randomization for all parameters"""
        # Check if any are enabled
        any_enabled = any(info.get('randomize', False) for info in self.parameters.values())
        
        # If any are enabled, disable all; otherwise enable all
        new_state = not any_enabled
        
        for name, param_info in self.parameters.items():
            param_info['randomize'] = new_state
            widget_data = self.param_widgets[name]
            widget_data['randomize_var'].set(new_state)
            self.update_parameter_visual_state(name)
            
        self.update_randomization_status()
        
    def save_randomization_profile(self):
        """Save current randomization settings to file"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Randomization Profile",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                profile_data = {}
                for name, param_info in self.parameters.items():
                    profile_data[name] = {
                        'randomize': param_info.get('randomize', False),
                        'rand_min': param_info.get('rand_min', param_info['min']),
                        'rand_max': param_info.get('rand_max', param_info['max'])
                    }
                    
                with open(filename, 'w') as f:
                    json.dump(profile_data, f, indent=2)
                    
                messagebox.showinfo("Success", f"Randomization profile saved to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profile: {str(e)}")
            
    def load_randomization_profile(self):
        """Load randomization settings from file"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Randomization Profile",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    profile_data = json.load(f)
                    
                for name, settings in profile_data.items():
                    if name in self.parameters:
                        param_info = self.parameters[name]
                        param_info['randomize'] = settings.get('randomize', False)
                        param_info['rand_min'] = settings.get('rand_min', param_info['min'])
                        param_info['rand_max'] = settings.get('rand_max', param_info['max'])
                        
                        # Update widget
                        widget_data = self.param_widgets[name]
                        widget_data['randomize_var'].set(param_info['randomize'])
                        widget_data['min_var'].set(param_info['rand_min'])
                        widget_data['max_var'].set(param_info['rand_max'])
                        self.update_parameter_visual_state(name)
                        
                self.update_randomization_status()
                messagebox.showinfo("Success", f"Randomization profile loaded from {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load profile: {str(e)}")
            
    def randomize_parameters(self):
        """Randomize only the parameters that have randomization enabled"""
        randomized_count = 0
        
        for name, widget_info in self.param_widgets.items():
            param_info = widget_info['info']
            
            # Only randomize if enabled
            if param_info.get('randomize', False):
                var = widget_info['var']
                
                # Use custom randomization range
                rand_min = param_info.get('rand_min', param_info['min'])
                rand_max = param_info.get('rand_max', param_info['max'])
                
                # Generate random value within custom range
                range_size = rand_max - rand_min
                random_value = rand_min + random.random() * range_size
                
                # Snap to step
                snapped = round(random_value / param_info['step']) * param_info['step']
                snapped = max(param_info['min'], min(param_info['max'], snapped))
                
                var.set(snapped)
                param_info['value'] = snapped
                widget_info['label'].config(text=f"{snapped:.3f}")
                randomized_count += 1
                
        if randomized_count == 0:
            messagebox.showwarning("No Randomization", "No parameters are enabled for randomization. Enable parameters using the checkboxes.")
        else:
            print(f"Randomized {randomized_count} parameters")
            
    def choose_output_directory(self):
        """Let user choose custom output directory"""
        directory = filedialog.askdirectory(title="Choose Output Directory")
        if directory:
            self.custom_output_dir = directory
            self.output_dir_var.set(f"Custom: {os.path.basename(directory)}")
        else:
            self.custom_output_dir = None
            self.output_dir_var.set("Auto-generated")
            
    def open_output_directory(self):
        """Open the current output directory in file explorer"""
        try:
            if hasattr(self.simulation, 'output_dir') and os.path.exists(self.simulation.output_dir):
                import subprocess
                import platform
                
                if platform.system() == "Windows":
                    subprocess.Popen(['explorer', self.simulation.output_dir])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(['open', self.simulation.output_dir])
                else:  # Linux
                    subprocess.Popen(['xdg-open', self.simulation.output_dir])
            else:
                messagebox.showwarning("No Directory", "No output directory available yet. Start a simulation first.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open directory: {e}")
            
    def get_current_parameters(self):
        """Get current parameter values"""
        return {name: info['value'] for name, info in self.parameters.items()}
        
    def start_simulation(self):
        """Start the simulation"""
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running")
            
            # Update status bar
            sim_id = int(time.time())
            self.sim_id_label.config(text=f"Simulation ID: {sim_id}")
            self.field_res_label.config(text=f"Field Resolution: {int(self.parameters['size']['value'])}³")
            self.sim_status_label.config(text="RUNNING", foreground="green")
            self.files_saved_label.config(text="Files saved: 0")
            
            # Clear metrics history
            self.metrics_history = []
            
            # Create simulation instance
            params = self.get_current_parameters()
            self.simulation = LilithSimulation(params, self.output_queue, self.custom_output_dir)
            
            # Update output directory display
            if hasattr(self.simulation, 'output_dir'):
                self.output_dir_var.set(f"Saving to: {os.path.basename(self.simulation.output_dir)}")
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self.simulation.run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
    def stop_simulation(self):
        """Stop the simulation"""
        if self.running:
            self.running = False
            if self.simulation:
                self.simulation.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped")
            self.sim_status_label.config(text="STOPPED", foreground="red")
            
    def reset_simulation(self):
        """Reset the simulation"""
        self.stop_simulation()
        self.step_label.config(text="Step: 0")
        self.metrics_history = []
        
        # Update status bar
        self.sim_id_label.config(text="Simulation ID: None")
        self.field_res_label.config(text="")
        self.sim_status_label.config(text="READY", foreground="blue")
        
        # Clear plots
        self.moll_fig.clear()
        self.ps_fig.clear()
        self.metrics_fig.clear()
        self.moll_canvas.draw()
        self.ps_canvas.draw()
        self.metrics_canvas.draw()
        
        # Reset metrics display
        for label in self.metrics_labels.values():
            label.config(text=label.cget('text').split(':')[0] + ": 0.000")
            
    def update_metrics_display(self, metrics):
        """Update the metrics display"""
        mapping = {
            'KL Divergence': 'kl_divergence',
            'Correlation': 'correlation', 
            'Entropy': 'entropy',
            'Observers': 'observer_count',
            'Shell Energy': 'shell_energy',
            'Coherence': 'coherence_index'
        }
        
        for display_name, metric_key in mapping.items():
            if metric_key in metrics:
                value = metrics[metric_key]
                if isinstance(value, (int, float)):
                    if display_name == 'Observers':
                        text = f"{display_name}: {int(value)}"
                    else:
                        text = f"{display_name}: {value:.4f}"
                    self.metrics_labels[display_name].config(text=text)
                    
    def update_mollweide_plot(self, projection_data):
        """Update the Mollweide projection plot"""
        self.moll_fig.clear()
        ax = self.moll_fig.add_subplot(111)
        
        try:
            # Handle different projection data formats
            if projection_data is not None and len(projection_data) > 0:
                # Try to create a simple 2D visualization of the projection
                if len(projection_data) == 12288:  # nside=64
                    side_len = 64
                elif len(projection_data) == 49152:  # nside=128  
                    side_len = 128
                elif len(projection_data) == 196608:  # nside=256
                    side_len = 256
                else:
                    # Default fallback
                    side_len = int(np.sqrt(len(projection_data) / 12))
                    if side_len < 32:
                        side_len = 32
                
                # Create a simplified 2D projection
                try:
                    # Reshape to approximate 2D grid
                    grid_size = min(128, side_len)
                    if len(projection_data) >= grid_size * grid_size:
                        data_2d = projection_data[:grid_size*grid_size].reshape((grid_size, grid_size))
                    else:
                        # Pad or truncate data
                        padded_data = np.zeros(grid_size * grid_size)
                        padded_data[:len(projection_data)] = projection_data
                        data_2d = padded_data.reshape((grid_size, grid_size))
                    
                    if np.max(data_2d) > np.min(data_2d):  # Check for non-zero variation
                        im = ax.imshow(np.log1p(np.abs(data_2d)), cmap='inferno', aspect='auto')
                        ax.set_title(f"Field Projection (Step {getattr(self.simulation, 'step', 0)})")
                        self.moll_fig.colorbar(im, ax=ax, shrink=0.6)
                        ax.set_xlabel('Longitude (projected)')
                        ax.set_ylabel('Latitude (projected)')
                    else:
                        ax.text(0.5, 0.5, 'No field variation yet...', 
                               transform=ax.transAxes, ha='center', va='center', fontsize=12)
                        ax.set_title(f"Field Projection (Step {getattr(self.simulation, 'step', 0)})")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Visualization Error:\n{str(e)}", 
                           transform=ax.transAxes, ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'Waiting for projection data...', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title("Field Projection")
                
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot Error:\n{str(e)}", 
                   transform=ax.transAxes, ha='center', va='center')
            
        self.moll_canvas.draw()
        
    def update_power_spectrum_plot(self, cl_data):
        """Update the power spectrum plot"""
        self.ps_fig.clear()
        ax = self.ps_fig.add_subplot(111)
        
        if cl_data is not None and len(cl_data) > 0:
            ell = np.arange(len(cl_data))
            ax.loglog(ell[1:], cl_data[1:], label='Simulation', color='orange')
            
            # Add Planck comparison if available
            if hasattr(self.simulation, 'planck_cl') and self.simulation.planck_cl is not None:
                planck_truncated = self.simulation.planck_cl[:len(cl_data)]
                ax.loglog(ell[1:len(planck_truncated)], planck_truncated[1:], 
                         label='Planck', linestyle='--', color='blue')
                         
            ax.set_xlabel('Multipole moment $\ell$')
            ax.set_ylabel('Cℓ')
            ax.set_title('Angular Power Spectrum')
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Waiting for data...', 
                   transform=ax.transAxes, ha='center', va='center')
                   
        self.ps_canvas.draw()
        
    def update_metrics_history_plot(self):
        """Update the metrics history plot"""
        if len(self.metrics_history) < 2:
            return
            
        self.metrics_fig.clear()
        
        # Create subplots for different metrics
        ax1 = self.metrics_fig.add_subplot(221)
        ax2 = self.metrics_fig.add_subplot(222)
        ax3 = self.metrics_fig.add_subplot(223)
        ax4 = self.metrics_fig.add_subplot(224)
        
        steps = [m['step'] for m in self.metrics_history]
        
        # KL Divergence
        kl_values = [m.get('kl_divergence', 0) for m in self.metrics_history]
        ax1.plot(steps, kl_values, 'r-', label='KL Divergence')
        ax1.set_title('KL Divergence vs Planck')
        ax1.set_ylabel('KL Divergence')
        ax1.grid(True)
        
        # Correlation
        corr_values = [m.get('correlation', 0) for m in self.metrics_history]
        ax2.plot(steps, corr_values, 'b-', label='Correlation')
        ax2.set_title('Correlation with Planck')
        ax2.set_ylabel('Correlation')
        ax2.grid(True)
        
        # Observer Count
        obs_values = [m.get('observer_count', 0) for m in self.metrics_history]
        ax3.plot(steps, obs_values, 'g-', label='Observers')
        ax3.set_title('Observer Population')
        ax3.set_ylabel('Observer Count')
        ax3.set_xlabel('Step')
        ax3.grid(True)
        
        # Shell Energy
        energy_values = [m.get('shell_energy', 0) for m in self.metrics_history]
        ax4.plot(steps, energy_values, 'm-', label='Shell Energy')
        ax4.set_title('Shell Energy')
        ax4.set_ylabel('Energy')
        ax4.set_xlabel('Step')
        ax4.grid(True)
        
        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw()
        
    def update_gui(self):
        """Main GUI update loop"""
        # Process output queue
        try:
            while True:
                msg_type, data = self.output_queue.get_nowait()
                
                if msg_type == 'metrics':
                    self.metrics_history.append(data)
                    self.update_metrics_display(data)
                    self.step_label.config(text=f"Step: {data['step']}")
                    self.update_metrics_history_plot()
                    
                elif msg_type == 'visualization':
                    if 'projection' in data:
                        self.update_mollweide_plot(data['projection'])
                    if 'power_spectrum' in data:
                        self.update_power_spectrum_plot(data['power_spectrum'])
                        
                elif msg_type == 'files_saved':
                    count = data.get('count', 0)
                    self.files_saved_label.config(text=f"Files saved: {count}")
                        
                elif msg_type == 'simulation_complete':
                    self.stop_simulation()
                    self.status_label.config(text="Status: Complete")
                    
        except queue.Empty:
            pass
            
        # Auto-randomize if enabled
        if self.auto_randomize.get() and self.running:
            if not hasattr(self, 'last_randomize_time'):
                self.last_randomize_time = time.time()
            elif time.time() - self.last_randomize_time > self.randomize_interval.get() / 1000.0:
                # Only auto-randomize if some parameters are enabled
                enabled_params = [name for name, info in self.parameters.items() if info.get('randomize', False)]
                if enabled_params:
                    self.randomize_parameters()
                    self.last_randomize_time = time.time()
                
        # Schedule next update
        self.root.after(50, self.update_gui)


def main():
    """Main application entry point"""
    print("Starting Lilith 1.0 GUI...")
    print(f"CuPy available: {CUPY_AVAILABLE}")
    
    # Check for required files
    fits_files = ["SMICA_CMB.FITS", "smica_cmb.fits", "COM_CMB_IQU-smica_1024_R2.02_full.fits"]
    fits_found = any(os.path.exists(f) for f in fits_files)
    
    cl_files = ["COM_PowerSpect_CMB-TT-full_R3.01.txt", "planck_2018_cls.txt"]
    cl_found = any(os.path.exists(f) for f in cl_files)
    
    print(f"Planck FITS file found: {fits_found}")
    print(f"Planck Cl file found: {cl_found}")
    
    if not fits_found and not cl_found:
        print("Warning: No Planck data files found. Simulation will run but without CMB comparison.")
        print("Expected files: SMICA_CMB.FITS or planck Cl text files")
    
    root = tk.Tk()
    app = LilithGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Shutting down...")
        if app.simulation:
            app.simulation.stop()
        root.quit()
    except Exception as e:
        print(f"Application error: {e}")
        if app.simulation:
            app.simulation.stop()
        root.quit()


if __name__ == "__main__":
    main()