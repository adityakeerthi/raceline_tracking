import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# Tuning parameters
LOOKAHEAD_MIN = 8.0
LOOKAHEAD_GAIN = 0.5
MAX_LATERAL_ACCEL = 8.0
BRAKE_DECEL = 14.0

# Controller state
raceline = None
velocity_profile = None
last_idx = 0

class PIDController:
    def __init__( self, kp, ki = 0.0 ):
        self.kp, self.ki = kp, ki
        self.integral = 0.0
    
    def compute( self, error, dt=0.1 ):
        self.integral = np.clip( self.integral + error * dt, -5.0, 5.0 )
        return self.kp * error + self.ki * self.integral

velocity_pid = PIDController( kp = 8.0, ki = 1.0 )
steering_pid = PIDController( kp = 10.0 )

def radius_from_triangle( pre, curr, post ):
    # Area calculation: https://en.wikipedia.org/wiki/Heron%27s_formula
    a = np.linalg.norm( post - curr )
    b = np.linalg.norm( post - pre )
    c = np.linalg.norm( curr - pre )
    
    # Semi-perimeter
    s = ( a + b + c ) / 2.0
    
    # Area using Heron's formula
    area_squared = s * ( s - a ) * ( s - b ) * ( s - c )
    
    if area_squared < 1e-12: 
        return np.inf 
    
    area = np.sqrt( area_squared )
    
    return ( a * b * c ) / ( 4.0 * area )

def compute_curvatures( path ):
    n = len( path )
    curvatures = np.zeros( n )

    for i in range( n ):
        pre = path[ ( i - 1) % n ]
        curr = path[ i ]
        post = path[ ( i + 1 ) % n ]
        
        R = radius_from_triangle( pre, curr, post )
        curvatures[ i ] = 1.0 / R if R < np.inf else 0.0

    return curvatures

def compute_velocity_profile( path, curvatures, max_vel ):
    # Just precompute desired velocity along the path
    # we need to account for the speed we maintain during
    # a curvature 
    n = len( path )
    vel = np.zeros( n )
    
    for i in range( n ):
        if curvatures[ i ] > 1e-4:
            # either go max vel or ideally, the velocity
            # note that curvatures[ i ] is the centripetal accel
            # so it's like a_lateral = v^2 * curvature; if we
            # solve for v, we get sqrt( a_lateral / curvature )
            vel[ i ] = min( np.sqrt( MAX_LATERAL_ACCEL / curvatures[ i ] ), max_vel )
        else:
            vel[ i ] = max_vel
    
    for i in range( n - 2, -1, -1 ):
        # how far we are from the next point, and braking early
        dist = np.linalg.norm( path[ ( i + 1 ) % n ] - path[ i ] )
        # consider v1^2 = v2^2 + 2ad, a kinemtic equation
        # consider a = BRAKE_DECEL, how much we can brake?
        vel[ i ] = min( 
            vel[ i ], 
            np.sqrt( vel[ ( i + 1 ) % n ] ** 2 + 2 * BRAKE_DECEL * dist )
        )
    
    return vel

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    max_steering_rate, max_accel = parameters[ 9 ], parameters[ 10 ]
    
    steering_rate = steering_pid.compute( desired[ 0 ] - state[ 2 ] )
    acceleration = velocity_pid.compute( desired[ 1 ] - state[ 3 ] )
    
    return np.array([
        np.clip( steering_rate, -max_steering_rate, max_steering_rate ),
        np.clip( acceleration, -max_accel, max_accel )
    ])


def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    global raceline, velocity_profile, last_idx
    
    if raceline is None:
        # precompute everything on first tick by looking ahead and precomputing
        # velocities and curvatures, and keeping track of the raceline
        raceline = racetrack.centerline.copy()
        curvatures = compute_curvatures( raceline )
        velocity_profile = compute_velocity_profile( raceline, curvatures, parameters[ 5 ] )
        last_idx = 0
    
    wheelbase, max_steering = parameters[ 0 ], parameters[ 4 ]
    pos, vel, heading = state[ 0: 2 ], max( state[ 3 ], 0.1 ), state[ 4 ]
    n = len( raceline )
    
    # Find closest point on raceline
    best_idx, best_dist = last_idx, np.inf
    for offset in range( -5, 40 ): # arbitrayr range
        idx = ( last_idx + offset ) % n
        dist = np.linalg.norm( pos - raceline[ idx ] )
        if dist < best_dist:
            best_dist, best_idx = dist, idx
    last_idx = best_idx
    
    # Find lookahead point
    lookahead = LOOKAHEAD_GAIN * vel + LOOKAHEAD_MIN
    acc_dist, idx = 0.0, best_idx
    while acc_dist < lookahead:
        # it isn't sufficient enough to look a magic constanty number of
        # points ahead the raceline
        # path points can be close or farther depending on the track
        # so do it off calculating the distance
        next_idx = ( idx + 1 ) % n
        acc_dist += np.linalg.norm( raceline[ next_idx ]  - raceline[ idx ] )
        idx = next_idx
    goal = raceline[ idx ]
    
    # Pure pursuit: transform goal to vehicle frame and compute steering
    # from tractor paper.
    # curvature = [ 2 * y ] / Lookahead*2
    dx, dy = goal[ 0 ] - pos[ 0 ], goal[ 1 ] - pos[ 1 ]
    local_x = dx * np.cos( heading ) + dy * np.sin( heading )
    local_y = -dx * np.sin( heading ) + dy * np.cos( heading )
    L_d = np.sqrt( local_x ** 2 + local_y ** 2 )
    
    if L_d > 0.1:
        curvature = 2.0 * local_y / ( L_d ** 2 )
        steering_angle = np.arctan( curvature * wheelbase )
        desired_steering = np.clip( steering_angle, -max_steering, max_steering )
    else:
        # just go straight?
        desired_steering = 0.0
    
    return np.array( [
        desired_steering, velocity_profile[ best_idx ] 
    ] )
