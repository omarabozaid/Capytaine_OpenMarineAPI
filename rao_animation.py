"""

Snippet that show how to catch wave data from openmarine api and use it for RAO animations
RAOs are computed from Capytaine
Ship is generated using ShipD code by Noah J. Bagazinski check his repo https://github.com/noahbagz

Author of this snippet : Dr.-Ing. Omar ELSAYED, 30.06.2021-MIT license
"""

from numpy import pi
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
from capytaine.ui.vtk import Animation
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from PIL import Image, ImageDraw, ImageFont
import numpy as np 

# Define the BEMSolver globally
bem_solver = cpt.BEMSolver()

# Function to get wave data from Open-Meteo API
def get_wave_data(latitude, longitude):
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
	retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
	openmeteo = openmeteo_requests.Client(session=retry_session)

	# Make sure all required weather variables are listed here
	url = "https://marine-api.open-meteo.com/v1/marine"
	params = {
		"latitude": latitude,
		"longitude": longitude,
		"hourly": ["wave_height", "wave_direction", "wave_period"]
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process the first location. Add a for-loop for multiple locations or weather models
	response = responses[0]

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_wave_height = hourly.Variables(0).ValuesAsNumpy()
	hourly_wave_direction = hourly.Variables(1).ValuesAsNumpy()
	hourly_wave_period = hourly.Variables(2).ValuesAsNumpy()

	hourly_data = {
		"date": pd.date_range(
			start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
			end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
			freq=pd.Timedelta(seconds=hourly.Interval()),
			inclusive="left"
		),
		"wave_height": hourly_wave_height,
		"wave_direction": hourly_wave_direction,
		"wave_period": hourly_wave_period
	}
	hourly_dataframe = pd.DataFrame(data=hourly_data)
	return hourly_dataframe

# Function to generate the boat in Capytaine
def generate_boat(filename):
	sphere = cpt.io.mesh_loaders.load_VTK(filename)
	boat = cpt.FloatingBody(
		mesh=sphere,
		dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)),
		center_of_mass=(0, 0, 0)
	)
	boat.inertia_matrix = boat.compute_rigid_body_inertia() / 10  # Artificially lower to have a more appealing animation
	boat.hydrostatic_stiffness = boat.immersed_part().compute_hydrostatic_stiffness()
	return boat

# Function to set up and run the animation in Capytaine
def setup_animation(body, fs, omega, wave_amplitude, wave_direction):
	global bem_solver  # Ensure bem_solver is available in this function
	# SOLVE BEM PROBLEMS
	radiation_problems = [cpt.RadiationProblem(omega=omega, body=body.immersed_part(), radiating_dof=dof) for dof in body.dofs]
	radiation_results = bem_solver.solve_all(radiation_problems)
	diffraction_problem = cpt.DiffractionProblem(omega=omega, body=body.immersed_part(), wave_direction=wave_direction)
	diffraction_result = bem_solver.solve(diffraction_problem)

	dataset = cpt.assemble_dataset(radiation_results + [diffraction_result])
	rao = cpt.post_pro.rao(dataset, wave_direction=wave_direction)

	# COMPUTE FREE SURFACE ELEVATION
	incoming_waves_elevation = airy_waves_free_surface_elevation(fs, diffraction_result)
	diffraction_elevation = bem_solver.compute_free_surface_elevation(fs, diffraction_result)

	radiation_elevations_per_dof = {res.radiating_dof: bem_solver.compute_free_surface_elevation(fs, res) for res in radiation_results}
	radiation_elevation = sum(rao.sel(omega=omega, radiating_dof=dof).data * radiation_elevations_per_dof[dof] for dof in body.dofs)

	# SET UP ANIMATION
	rao_faces_motion = sum(rao.sel(omega=omega, radiating_dof=dof).data * body.dofs[dof] for dof in body.dofs)

	animation = Animation(loop_duration=6 * pi / omega)
	animation.add_body(body, faces_motion=wave_amplitude * rao_faces_motion)
	animation.add_free_surface(fs, wave_amplitude * (incoming_waves_elevation + diffraction_elevation + radiation_elevation))
	return animation

# Function to plot wave data
def plot_wave_data(wave_height, wave_period, wave_direction, index):
	plt.figure(figsize=(10, 4))

	# Plot wave height
	plt.subplot(1, 3, 1)
	plt.plot(wave_height, 'b')
	plt.title('Wave Height (m)')
	plt.xlabel('Time (hours)')
	plt.ylabel('Height (m)')

	# Plot wave period
	plt.subplot(1, 3, 2)
	plt.plot(wave_period, 'g')
	plt.title('Wave Period (s)')
	plt.xlabel('Time (hours)')
	plt.ylabel('Period (s)')

	# Plot wave direction
	plt.subplot(1, 3, 3)
	plt.plot(wave_direction, 'r')
	plt.title('Wave Direction (°)')
	plt.xlabel('Time (hours)')
	plt.ylabel('Direction (°)')

	plt.tight_layout()
	plt.savefig(f"wave_data_{index}.png")
	plt.close()

# Function to create a text image using PIL
def create_text_image(text, fontsize=24, text_color='black', background_color='white', size=(800, 100)):
    img = Image.new('RGB', size, color=background_color)  # Create white background image
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", fontsize)
    text_size = draw.textbbox((0, 0), text, font=font)
    text_position = ((size[0] - text_size[2]) // 2, (size[1] - text_size[3]) // 2)
    draw.text(text_position, text, font=font, fill=text_color)
    return np.array(img)

if __name__ == '__main__':
	# Get wave data
	latitude = 43.3026
	longitude = 5.3691
	wave_data = get_wave_data(latitude, longitude)

	filename = "vessel.vtk"
	body = generate_boat(filename)
	fs = cpt.FreeSurface(x_range=(-35, 35), y_range=(-35, 35), nx=100, ny=100)

	# Generate and save an animation for the last 4 hours
	last_hours = 6
	animation_clips = []

	for index, row in wave_data.iloc[-last_hours:].iterrows():
		wave_height = row["wave_height"]
		wave_direction = row["wave_direction"]
		wave_period = row["wave_period"]

		# Calculate omega from wave period
		omega = 2 * pi / wave_period

		# Set up and run the animation
		anim = setup_animation(body, fs, omega=omega, wave_amplitude=wave_height, wave_direction=wave_direction)
		anim.run(camera_position=(35, 35, 100), resolution=(800, 600))
		animation_file = f"animated_boat_{index}.ogv"
		anim.save(animation_file, camera_position=(70, 70, 100), resolution=(800, 600))

		# Create a MoviePy video clip
		clip = mpy.VideoFileClip(animation_file)

		# Create text image for wave data
		text = f"Wave Height: {wave_height:.3f} m\nWave Period: {wave_period:.3f} s\nWave Direction: {wave_direction:.3f}°"
		text_img = create_text_image(text)
		text_clip = mpy.ImageClip(text_img).set_duration(clip.duration).set_position(('center', 'bottom'))

		# Overlay text on video
		clip = mpy.CompositeVideoClip([clip, text_clip])

		# Append the clip to the list
		animation_clips.append(clip)

		# Plot and save wave data
		plot_wave_data(wave_data["wave_height"], wave_data["wave_period"], wave_data["wave_direction"], index)

		print(f"Saved animation and wave data for hour {index}")

	# Concatenate all clips into one final video
	final_clip = mpy.concatenate_videoclips(animation_clips)
	final_clip.write_videofile("combined_animated_boat.mp4", fps=24)
