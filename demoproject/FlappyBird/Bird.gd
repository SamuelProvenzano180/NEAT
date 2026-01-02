extends Area2D

var randNum = RandomNumberGenerator.new()
var velocity = Vector2.ZERO

# --- CONVERTED CONSTANTS ---
# Old Gravity: 0.4 (per frame) * 60 * 60 = ~1500.0
# Old Jump: -8.0 (per frame) * 60 = -480.0
# Old Max: 12.0 (per frame) * 60 = 720.0

var GRAVITY = 1500.0
var JUMP_FORCE = -480.0
var MAX_SPEED = 720.0

var hit_pipe = false
var dead = false

func _ready() -> void:
	randNum.randomize()
	var color = Color.WHITE
	
	var color_id = randNum.randi_range(0, 8)
	
	if color_id == 0:
		color = Color.RED
	elif color_id == 1:
		color = Color.BLUE
	elif color_id == 2:
		color = Color.GREEN
	elif color_id == 3:
		color = Color.PURPLE
	elif color_id == 4:
		color = Color.YELLOW
	elif color_id == 5:
		color = Color.CYAN
	elif color_id == 6:
		color = Color.CORAL
	elif color_id == 7:
		color = Color.DARK_SEA_GREEN
	elif color_id == 8:
		color = Color.HOT_PINK
	
	$MeshInstance2D.modulate = color

func _physics_process(delta: float) -> void:
	# 1. Apply Gravity (scaled by delta)
	velocity.y += GRAVITY * delta
	
	# 2. Clamp (Velocity is now in pixels/sec, so limits are higher)
	velocity.y = clamp(velocity.y, -MAX_SPEED, MAX_SPEED)
	
	if !dead:
		# 3. Move (scaled by delta)
		global_position += velocity * delta
	else:
		visible = false

func jump():
	# Set instant upward velocity
	velocity.y = JUMP_FORCE

func _on_area_entered(area: Area2D) -> void:
	# Check for "Pipe" group or the specific Pipe area name
	# (Ensure your Pipe scene's Area2D is actually in the group "pipe"!)
	if area.is_in_group("pipe"): 
		hit_pipe = true
