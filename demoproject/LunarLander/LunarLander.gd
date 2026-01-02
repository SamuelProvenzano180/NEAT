extends Node2D #On initialization, set the champion network to the first network created. It being null may cause some errors

var neat = NEATAgent.new()
var net = NetworkAgent.new()
var is_using_import_data = false

@onready var ShipScene = preload("res://LunarLander/Ship.tscn") # Make sure this matches your file name!

# --- PHYSICS CONSTANTS ---
const GRAVITY = 1300.0
const ENGINE_ACCEL = 2000.0
const TORQUE_ACCEL = 40.0   # Increased torque to help them correct tilt
const ANG_DAMP = 2.0        # Loosened damping so they don't feel like they are in molasses
const MAX_SPEED = 600.0
const MAX_ANGVEL = 8.0

var rng = RandomNumberGenerator.new()
var generation = 1

# Arrays to store data for each ship
var scores = []
var playing = []

# SETTINGS
var population_size = 250    # Keep this reasonable for your CPU
var time_left = 4.0
var inputs = 8              # Must match get_ship_state
var outputs = 2             # Throttle, Turn

var test_best_network = false
var test = false

func _ready():
	rng.randomize()
	
	# Initialize NEAT with 8 Inputs, 2 Outputs
	
	# High weight mutation to find "handling", Low structural mutation to keep brains stable
	#neat.initialize_population(inputs, outputs, population_size, "tanh", "tanh", 15)
	#neat.set_mutation_rates(0.85, 0.13, 0.1, 0.07)
	#neat.set_stagnation_limit(80)
	#neat.set_connection_size_limit(50)
	
	var import_data = [9, 2, 3, 3, [0, 10, 3.28829145431519], [1, 9, -4.54019737243652], [4, 9, -1.86439716815948], [4, 10, -1.46769046783447], [5, 9, 2.6042423248291], [6, 9, -2.82958269119263], [6, 10, -3.92524337768555], [7, 10, -1.55836725234985], [7, 15, 1.82845854759216], [11, 10, 4.8511176109314], [12, 10, 0.56241208314896], [12, 13, 3.39294171333313], [13, 14, 3.67162823677063], [13, 15, 1.43402576446533], [14, 15, 1.65369081497192], [15, 9, -3.24449777603149]]

	
	neat.import_template(import_data, population_size, 10)
	neat.set_mutation_rates(0.8, 0.1, 0.05, 0.02)
	neat.set_stagnation_limit(70)
	neat.set_connection_size_limit(100)
	
	net.initialize_agent(import_data)
	
	if is_using_import_data:
		reset_import_episode()
	else:
		reset_episode()

func _process(_delta):
	# Manual Reset for panic situations
	if Input.is_action_just_pressed("ui_accept"): 
		test_best_network = !test_best_network

func _physics_process(delta):
	if Input.is_action_just_pressed("ui_left"):
		print(neat.extract_champion_data())
	
	if Input.is_action_just_pressed("p"):
		neat.force_champion_reset()
		print("CHAMPION ERASED")
	
	if is_using_import_data:
		import_test(delta)
	else:
		if test:
			test_(delta)
		else:
			train(delta)

func import_test(delta):#[9, 2, 3, 3, [0, 10, 2.71752381324768], [4, 10, -1.33246469497681], [6, 10, -1.94952845573425], [7, 10, -0.71123695373535]]
	time_left -= delta
	
	var ship = $Ships.get_child(0)
	
	# --- 1. GET INPUTS & NETWORK GUESS ---
	var current_state = get_ship_state(ship)
	
	var guess = net.guess(current_state)
	
	# --- 2. INTERPRET OUTPUTS ---
	# Output 0 (Throttle): Network gives -1 to 1. We map to 0 to 1.
	var throttle = (guess[0] + 1.0) / 2.0 
	var turn = guess[1] # Tanh is naturally -1 to 1. Perfect for torque.
	
	update_physics(delta, ship, throttle, turn)
	
	ship.change_particles(throttle)
	
	var is_dead = false
		# Rule B: INSTANT KILL if Out of Bounds
	if abs(ship.global_position.x) > 700.0 or ship.global_position.y < -400.0:
		is_dead = true
		
	# Rule C: GROUND INTERACTION
	if ship.global_position.y > 324.0: # Ground Level
		is_dead = true
	
	if time_left <= 0.0:
		is_dead = true
		
	# Next Generation Logic
	if is_dead == true:
		reset_import_episode()
	

func test_(delta):
	time_left -= delta
	
	var ship = $Ships.get_child(0)
	
	# --- 1. GET INPUTS & NETWORK GUESS ---
	var current_state = get_ship_state(ship)
	
	var guess = neat.get_champion_guess(current_state)
	
	# --- 2. INTERPRET OUTPUTS ---
	# Output 0 (Throttle): Network gives -1 to 1. We map to 0 to 1.
	var throttle = (guess[0] + 1.0) / 2.0 
	var turn = guess[1] # Tanh is naturally -1 to 1. Perfect for torque.
	
	update_physics(delta, ship, throttle, turn)
	
	ship.change_particles(throttle)
	
		# Rule B: INSTANT KILL if Out of Bounds
	if abs(ship.global_position.x) > 700.0 or ship.global_position.y < -400.0:
		playing[0] = false
		
	# Rule C: GROUND INTERACTION
	if ship.global_position.y > 324.0: # Ground Level
		playing[0] = false
	
	if time_left <= 0.0:
		playing[0] = false
		
	# Next Generation Logic
	if playing[0] == false:
		reset_episode()

func train(delta):
	
	time_left -= delta
	
	var active_ships = 0
	
	# Safety: Ensure we don't access children if they don't exist
	var ship_count = $Ships.get_child_count()
	
	for i in population_size:
		
		if i >= ship_count: break # Safety break
		
		# If this agent is already dead/finished, skip it
		if playing[i] == false:
			continue
			
		active_ships += 1
		var ship = $Ships.get_child(i)
		
		# --- 1. GET INPUTS & NETWORK GUESS ---
		var current_state = get_ship_state(ship)
		
		# SAFETY CHECK: If inputs are NaN
		# CHANGED: Don't continue without setting fitness!
		if is_nan(current_state[0]): 
			playing[i] = false
			neat.set_network_fitness(i, 0.01) # SUBMIT MINIMUM SCORE BEFORE CONTINUING
			continue
		
		var guess = neat.get_network_guess(i, current_state)
		
		# --- 2. INTERPRET OUTPUTS ---
		var throttle = (guess[0] + 1.0) / 2.0 
		var turn = guess[1] 
		
		ship.change_particles(throttle)
		update_physics(delta, ship, throttle, turn)
		
		# FLAG TO TRACK DEATH THIS FRAME
		var just_died = false
		
		if abs(ship.rotation_degrees) > 90:
			just_died = true
			
			# CALCULATE SCORE ANYWAY
			var dist_to_flag = abs(ship.global_position.x - $Flag.global_position.x)
			
			# We use a wider divisor (1500) so even far away ships get > 0
			var proximity = clamp(1.0 - (dist_to_flag / 1500.0), 0.0, 1.0)
			
			# Give them a small partial score (e.g. max 2.0 pts)
			# This rewards trying to get close, even if you fail the landing.
			scores[i] = proximity * 2.0 

		# Rule B: Out of Bounds (Keep this 0.0, we don't want them flying off screen)
		elif abs(ship.global_position.x) > 700.0 or ship.global_position.y < -400.0:
			scores[i] = 0.0
			just_died = true
			
		# Rule C: GROUND INTERACTION
		elif ship.global_position.y > 324.0: 
			just_died = true
			
			var impact_speed = ship.velocity.length()
			var dist_to_flag = abs(ship.global_position.x - $Flag.global_position.x)
			
			if is_nan(dist_to_flag) or is_inf(dist_to_flag): dist_to_flag = 10000.0
			
			# 1. Calculate Proximity
			var linear_proximity = clamp(1.0 - (dist_to_flag / 800.0), 0.0, 1.0)
			var proximity_score = linear_proximity * linear_proximity
			
			
			# --- THE FIX: LANDING ZONE CHECK ---
			# Define a "Pad Radius" (e.g., 80 pixels around the flag)
			var landing_radius = 80.0
			
			if dist_to_flag > landing_radius:
				# MISSED THE PAD!
				# Touching the ground here is no better than dying in the air.
				# We give them the same small proximity reward (~5.0 max) so they still want to get close.
				scores[i] = proximity_score * 5.0
				
			else:
				# HIT THE PAD! (Now we care about speed/angle)
				var is_soft = impact_speed < 120.0
				var is_upright = abs(ship.rotation_degrees) < 30.0
				
				if not is_upright: is_soft = false 
				
				if is_soft and is_upright:
					# 1. BASE REWARD (Success)
					var base_score = 200.0
					
					# 2. EFFICIENCY BONUS (Time is Money)
					var time_bonus = time_left * 10.0 
					
					# 3. PRECISION BONUS (Bullseye)
					# 1.0 = Dead Center, 0.0 = Edge of pad
					var precision_factor = 1.0 - (dist_to_flag / landing_radius)
					var precision_bonus = precision_factor * 50.0
					
					# 4. STABILITY BONUS (Butter Landing)
					# 1.0 = Perfectly Flat, 0.0 = Tilted 30 degrees
					var tilt_factor = 1.0 - (abs(ship.rotation_degrees) / 30.0)
					var stability_bonus = tilt_factor * 50.0

					# 5. SOFTNESS BONUS (The Feather Touch) [NEW]
					# We verify speed is < 120.0 (checked by is_soft), so we normalize 0-120.
					# Speed 0.0   -> Factor 1.0 (Max Bonus)
					# Speed 60.0  -> Factor 0.5
					# Speed 119.0 -> Factor 0.0 (Min Bonus)
					var impact_factor = 1.0 - (impact_speed / 120.0)
					var softness_bonus = impact_factor * 100.0 

					# FINAL SCORE
					# Now the AI can always squeeze out a few more points by slowing down further.
					scores[i] = base_score + time_bonus + precision_bonus + stability_bonus + softness_bonus
	
		# Rule D: TIMEOUT
		elif time_left <= 0.0:
			just_died = true
			active_ships -= 1 # Safety for zombie frame
			
			var dist_to_flag = abs(ship.global_position.x - $Flag.global_position.x)
			if is_nan(dist_to_flag) or is_inf(dist_to_flag): dist_to_flag = 10000.0 # Safety
			scores[i] = clamp(1.0 - (dist_to_flag / 1152.0), 0.0, 1.0) * 5.0

		# --- 4. SUBMIT FITNESS IF DEAD ---
		if just_died:
			playing[i] = false
			
			# Safety Clamp: NEAT breaks if fitness is 0 or negative
			if is_nan(scores[i]): scores[i] = 0.01
			if scores[i] < 0.01: scores[i] = 0.01
			
			scores[i] += 0.02
			
			# NOW THIS LINE ALWAYS RUNS FOR DEAD SHIPS
			neat.set_network_fitness(i, scores[i])
	
	if active_ships <= 0:
		start_next_generation()

func start_next_generation():
	if !test:
		neat.next_generation()
		generation += 1
	
	print("Gen: " + str(generation) + " | Best: " + str(neat.get_champion_fitness()), " | Conn: ", neat.get_champion_connection_count())
	
	# Call reset deferred to ensure physics frame is done
	call_deferred("reset_episode")

func update_physics(delta, ship, throttle, turn):
	ship.velocity.y += GRAVITY * delta
	
	var thrust_dir = Vector2.UP.rotated(ship.rotation)
	ship.velocity += thrust_dir * (throttle * ENGINE_ACCEL * delta)
	
	ship.angular_velocity += turn * TORQUE_ACCEL * delta
	ship.angular_velocity *= exp(-ANG_DAMP * delta)

	if ship.velocity.length() > MAX_SPEED:
		ship.velocity = ship.velocity.normalized() * MAX_SPEED
		
	ship.angular_velocity = clamp(ship.angular_velocity, -MAX_ANGVEL, MAX_ANGVEL)

	ship.rotation += ship.angular_velocity * delta
	ship.global_position += ship.velocity * delta

func reset_episode():
	
	Engine.time_scale = 7.0
	
	for child in $Ships.get_children():
		child.queue_free()
	#
	## 2. Reset Logic Arrays
	scores.clear()
	playing.clear()
	time_left = 4.0
	#
	## 3. Randomize Map
	var flag_x = rng.randf_range(-400.0, 400.0)
	$Flag.global_position = Vector2(flag_x, 324.0)
	
	var ship_amount = population_size
	test = test_best_network
	
	if test:
		ship_amount = 1
	
	for i in ship_amount:
		var ship = ShipScene.instantiate()
		$Ships.add_child(ship)
		
		# Spawn High
		
		ship.global_position = Vector2(rng.randf_range(-400.0, 400.0), -200.0)
		
		# IMPORTANT: Random Tilt/Velocity to force them to learn control
		ship.rotation = rng.randf_range(-0.5, 0.5) # Radians (approx -30 to 30 deg)
		ship.velocity = Vector2(rng.randf_range(-50, 50), rng.randf_range(-50, 50))
		ship.angular_velocity = 0.0
		
		# Initialize State
		scores.append(0.0)
		playing.append(true)

func reset_import_episode():
	Engine.time_scale = 1.0
	
	for child in $Ships.get_children():
		child.queue_free()
	time_left = 12.0
	var flag_x = rng.randf_range(-400.0, 400.0)
	$Flag.global_position = Vector2(flag_x, 324.0)

	var ship = ShipScene.instantiate()
	$Ships.add_child(ship)
	
	# Spawn High
	
	ship.global_position = Vector2(rng.randf_range(-400.0, 400.0), -200.0)
	
	ship.rotation = rng.randf_range(-0.5, 0.5) # Radians (approx -30 to 30 deg)
	ship.velocity = Vector2(rng.randf_range(-50, 50), rng.randf_range(-50, 50))
	ship.angular_velocity = 0.0


func get_ship_state(ship):
	# 8 INPUTS TOTAL
	
	# 1. Relative Vector to Flag (Guidance)
	var diff_x = ($Flag.global_position.x - ship.global_position.x) / 1152.0
	var diff_y = ($Flag.global_position.y - ship.global_position.y) / 648.0
	
	# 2. Wall Sensors (Absolute Position)
	var wall_x = clamp(ship.global_position.x / 576.0, -1.0, 1.0)
	var wall_y = clamp(ship.global_position.y / 324.0, -1.0, 1.0)
	
	# 3. Physics State
	var vel_x = clamp(ship.velocity.x / MAX_SPEED, -1.0, 1.0)
	var vel_y = clamp(ship.velocity.y / MAX_SPEED, -1.0, 1.0)
	var rot = clamp(ship.rotation / 3.14159, -1.0, 1.0) # Normalized Radians
	var ang_vel = clamp(ship.angular_velocity / MAX_ANGVEL, -1.0, 1.0)
	
	return PackedFloat32Array([diff_x, diff_y, wall_x, wall_y, vel_x, vel_y, rot, ang_vel])
