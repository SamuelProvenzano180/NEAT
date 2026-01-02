extends Area2D

var randNum = RandomNumberGenerator.new()
var hole_size = 200.0
var speed = 600.0 # Increased for delta (Pixels/Sec)

var pipe_hole = 0.0

func _ready() -> void:
	randNum.randomize()
	
	# --- 1. Fix the Shared Shape Bug ---
	# We duplicate the shapes so modifying one doesn't affect the others
	$CollisionShape2D.shape = $CollisionShape2D.shape.duplicate()
	$CollisionShape2D2.shape = $CollisionShape2D2.shape.duplicate()
	
	var pipe_rand = hole_size / 2.0 + 10.0
	var hole_pos = randNum.randf_range(pipe_rand, 648.0 - pipe_rand)
	
	# Store this so the Bird can see it in 'get_bird_state'
	pipe_hole = hole_pos 
	
	var top_pipe_size = hole_pos - hole_size / 2.0
	var bottom_pipe_position = hole_pos + hole_size / 2.0
	
	# Visuals
	$ColorRect.size.y = top_pipe_size
	$ColorRect2.global_position.y = bottom_pipe_position
	var bottom_pipe_size = 648.0 - $ColorRect2.global_position.y
	$ColorRect2.size.y = bottom_pipe_size
	
	# Physics
	$CollisionShape2D.global_position.y = top_pipe_size / 2.0
	$CollisionShape2D.shape.size.y = top_pipe_size
	
	$CollisionShape2D2.global_position.y = bottom_pipe_position + (648 - bottom_pipe_position) / 2.0
	$CollisionShape2D2.shape.size.y = bottom_pipe_size

func _physics_process(delta: float) -> void:
	# --- 2. Use Delta for consistent speed ---
	global_position.x -= speed * delta
	
	if global_position.x <= -100: # Give it a buffer before deleting
		queue_free()
