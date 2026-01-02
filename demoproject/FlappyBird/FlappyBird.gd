extends Node2D

@onready var PipeScene = preload("res://FlappyBird/Pipe.tscn")
@onready var BirdScene = preload("res://FlappyBird/Bird.tscn")

var neat = NEATAgent.new()
var rng = RandomNumberGenerator.new()
var generation = 1

# Arrays
var scores = []
var playing = []

# SETTINGS
var population_size = 200
var inputs = 4 
var outputs = 1 

# Spawning Logic
var pipe_dist_x = 500.0  # Distance between pipes (in pixels)
var spawn_x = 1200.0     # Where pipes spawn

var test_best_network = false
var test = false

func _ready():
	rng.randomize()
	neat.initialize_population(inputs, outputs, population_size, "tanh", "tanh", 5)
	neat.set_mutation_rates(0.9, 0.1, 0.06, 0.04)
	neat.set_connection_size_limit(25)
	reset_episode()

func _physics_process(delta):
	if Input.is_action_just_pressed("ui_left"):
		print(neat.extract_champion_data())
	
	# --- 1. ROBUST SPAWNING LOGIC (Distance Based) ---
	# This ensures pipes never bunch up, even at 100x speed.
	var last_pipe = null
	if $Pipes.get_child_count() > 0:
		last_pipe = $Pipes.get_child($Pipes.get_child_count() - 1)
		
	# Spawn if no pipes exist OR the last pipe has moved far enough left
	if last_pipe == null or (spawn_x - last_pipe.global_position.x) > pipe_dist_x:
		spawn_new_pipe()
	
	# --- 2. TRAIN OR TEST ---
	if test:
		test_loop(delta)
	else:
		train(delta)

func spawn_new_pipe():
	var pipe = PipeScene.instantiate()
	$Pipes.add_child(pipe)
	pipe.global_position.x = spawn_x
	# Note: Your Pipe.gd _ready() handles the random hole generation automatically.

func train(delta):
	var active_birds = 0
	var bird_count = $Birds.get_child_count()
	var target_pipe = get_next_pipe()
	
	for i in population_size:
		if i >= bird_count: break
		
		if playing[i] == false: continue
			
		active_birds += 1 # Count them as alive initially
		var bird = $Birds.get_child(i)
		
		# --- A. GET INPUTS ---
		var current_state = get_bird_state(bird, target_pipe)
		
		# FIX 1: Handle NaN death properly
		if is_nan(current_state[0]):
			playing[i] = false
			neat.set_network_fitness(i, 0.01)
			active_birds -= 1 # <--- SUBTRACT HERE
			continue
		
		# --- B. NETWORK DECISION ---
		var guess = neat.get_network_guess(i, current_state)
		if guess[0] > 0.0:
			bird.jump() 
		
		# --- C. SCORING ---
		scores[i] += delta
		if target_pipe != null:
			var gap_y = target_pipe.pipe_hole 
			var dist_y = abs(bird.global_position.y - gap_y)
			var accuracy = clamp(1.0 - (dist_y / 200.0), 0.0, 1.0)
			scores[i] += accuracy * delta * 3.0
		
		# --- D. DEATH CHECKS ---
		var just_died = false

		if bird.global_position.y > 648.0 or bird.global_position.y < 0.0:
			just_died = true
		if bird.hit_pipe == true:
			just_died = true
		
		# --- E. SUBMIT FITNESS ---
		if just_died:
			playing[i] = false
			bird.visible = false
			
			if is_nan(scores[i]): scores[i] = 0.01
			if scores[i] < 0.01: scores[i] = 0.01
			
			neat.set_network_fitness(i, scores[i])
			
			# FIX 2: SUBTRACT HERE
			# If they die THIS frame, remove them from the count immediately.
			# This triggers the reset logic instantly at the bottom of the function.
			active_birds -= 1 

	# --- NEXT GENERATION ---
	# Now this will accurately trigger even if the last bird died 1 millisecond ago
	if active_birds <= 0:
		if !test:
			neat.next_generation()
			generation += 1
		print("Gen: " + str(generation) + " | Best: " + str(neat.get_champion_fitness()) + " | Conn: " + str(neat.get_champion_connection_count()))
		reset_episode()

# Helper to find the correct pipe
func get_next_pipe():
	var closest_pipe = null
	var closest_dist = 99999.0
	
	for pipe in $Pipes.get_children():
		# Only look at pipes that are in front of the bird (x > 40)
		# (Birds are at x=80, pipe width is approx 80, so 40 is safe center)
		if pipe.global_position.x > 40.0:
			if pipe.global_position.x < closest_dist:
				closest_dist = pipe.global_position.x
				closest_pipe = pipe
	
	return closest_pipe

func get_bird_state(bird, target_pipe):
	# 1. Bird Y (Normalized 0.0 to 1.0)
	var bird_y = clamp(bird.global_position.y / 648.0, 0.0, 1.0)
	
	# 2. Bird Velocity (Normalized -1.0 to 1.0)
	# Assuming max velocity is around 800
	var bird_vel = clamp(bird.velocity.y / 800.0, -1.0, 1.0)
	
	var dist_to_pipe = 1.0 # Default "Far away"
	var pipe_gap_y = 0.5   # Default "Middle"
	
	if target_pipe != null:
		# 3. Distance to Pipe X (Normalized 0.0 to 1.0)
		dist_to_pipe = clamp((target_pipe.global_position.x - bird.global_position.x) / 600.0, 0.0, 1.0)
		
		# 4. Pipe Gap Y (Normalized 0.0 to 1.0)
		# USES 'pipe_hole' VARIABLE FROM PIPE SCRIPT
		pipe_gap_y = clamp(target_pipe.pipe_hole / 648.0, 0.0, 1.0)
	
	return PackedFloat32Array([bird_y, bird_vel, dist_to_pipe, pipe_gap_y])

func reset_episode():
	# Speed up training
	Engine.time_scale = 1.0 
	
	# Clear logic arrays
	scores.clear()
	playing.clear()
	
	# Clear Scene
	for child in $Birds.get_children(): child.queue_free()
	for child in $Pipes.get_children(): child.queue_free()
	
	var count = 1 if test else population_size
	
	for i in count:
		var bird = BirdScene.instantiate()
		$Birds.add_child(bird)
		bird.global_position = Vector2(80.0, 324.0)
		scores.append(0.0)
		playing.append(true)

func test_loop(_delta):
	# Simple visualization loop for best network
	if $Birds.get_child_count() == 0: return
	
	var bird = $Birds.get_child(0)
	var target_pipe = get_next_pipe()
	var state = get_bird_state(bird, target_pipe)
	var guess = neat.get_champion_guess(state)
	
	if guess[0] > 0.0:
		bird.jump()
		
	if bird.global_position.y > 648 or bird.global_position.y < 0 or bird.hit_pipe:
		reset_episode()
