extends Node2D

var velocity = Vector2.ZERO
var angular_velocity = 0.0

func _ready() -> void:
	set_color(randi_range(0, 8))

func change_particles(throttle):
	var emit_amount = clamp(throttle * 50.0, 1, 50)
	$GPUParticles2D.amount = emit_amount

func set_color(color_id):
	var color = Color.WHITE
	
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
