"""
SUMARO SIH 2026 Solution Video Frame Renderer & Video Producer.
Renders all 11 scenes (2,700 frames at 30 fps = 90.0s) and pipes them
directly to FFmpeg with the master audio mix.
"""

import os
import sys
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

# Paths
AUDIO_MIX = "video_assets/audio/final_audio_mix.wav"
OUTPUT_VIDEO = "reports/videos/sumaro_sih_2026_solution_video.mp4"
ROOT_COPY = "sumaro_sih_2026_solution_video.mp4"

# Dimensions & Framerate
WIDTH = 1920
HEIGHT = 1080
FPS = 30
TOTAL_FRAMES = 2700  # 90.0s * 30 fps

# Windows Fonts
FONTS_DIR = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
FONT_REGULAR = os.path.join(FONTS_DIR, 'segoeui.ttf')
FONT_BOLD = os.path.join(FONTS_DIR, 'segoeuib.ttf')
FONT_CODE = os.path.join(FONTS_DIR, 'consola.ttf')

f_title = ImageFont.truetype(FONT_BOLD, 54)
f_sub = ImageFont.truetype(FONT_REGULAR, 28)
f_card_title = ImageFont.truetype(FONT_BOLD, 32)
f_body = ImageFont.truetype(FONT_REGULAR, 24)
f_body_bold = ImageFont.truetype(FONT_BOLD, 24)
f_mono = ImageFont.truetype(FONT_CODE, 20)
f_large = ImageFont.truetype(FONT_BOLD, 68)
f_hud = ImageFont.truetype(FONT_BOLD, 22)
f_badge = ImageFont.truetype(FONT_BOLD, 18)

# Color Palette (Hex to RGB)
C_BG_DARK = (11, 16, 27)
C_CARD_BG = (17, 24, 39)
C_CARD_BORDER = (31, 41, 55)
C_TEXT_WHITE = (249, 250, 251)
C_TEXT_MUTED = (156, 163, 175)
C_BLUE = (37, 99, 235)
C_CYAN = (6, 182, 212)
C_GREEN = (16, 185, 129)
C_RED = (239, 68, 68)
C_AMBER = (245, 158, 11)
C_PURPLE = (139, 92, 246)

# Load real UI screenshots if available for phone screens
UI_IMG_1 = None
UI_IMG_2 = None
try:
    p1 = r"C:\Users\shubh\.gemini\antigravity\brain\df126545-dac8-4187-8306-ab473e99a6b5\.user_uploaded\media_1790682074433.png"
    p2 = r"C:\Users\shubh\.gemini\antigravity\brain\df126545-dac8-4187-8306-ab473e99a6b5\.user_uploaded\media_1790682081504.png"
    if os.path.exists(p1):
        UI_IMG_1 = Image.open(p1).convert("RGBA")
    if os.path.exists(p2):
        UI_IMG_2 = Image.open(p2).convert("RGBA")
except Exception as e:
    pass

# Helper drawing routines
def draw_gradient_rect(draw, x0, y0, x1, y1, color_top, color_bottom):
    # Vertical gradient
    for y in range(y0, y1):
        alpha = (y - y0) / max(1, (y1 - y0))
        r = int(color_top[0] * (1 - alpha) + color_bottom[0] * alpha)
        g = int(color_top[1] * (1 - alpha) + color_bottom[1] * alpha)
        b = int(color_top[2] * (1 - alpha) + color_bottom[2] * alpha)
        draw.line([(x0, y), (x1, y)], fill=(r, g, b))

def draw_header_banner(draw, title_text, category_text, badge_color=C_BLUE):
    # Top banner background
    draw.rectangle([(0, 0), (WIDTH, 90)], fill=(15, 23, 42))
    draw.line([(0, 90), (WIDTH, 90)], fill=(30, 41, 59), width=2)
    
    # Category badge
    bw = draw.textlength(category_text, font=f_badge) + 24
    draw.rounded_rectangle([(60, 26), (60 + bw, 62)], radius=6, fill=badge_color)
    draw.text((72, 32), category_text, font=f_badge, fill=(255, 255, 255))
    
    # Title
    draw.text((60 + bw + 24, 25), title_text, font=f_card_title, fill=C_TEXT_WHITE)
    
    # Project watermark
    wm = "SUMARO • SIH 2026"
    draw.text((WIDTH - 240, 32), wm, font=f_badge, fill=C_CYAN)

def draw_road_perspective(draw, t, in_tunnel=False, tunnel_alpha=0.0):
    # Vanishing point
    vp_x = WIDTH // 2
    vp_y = 480
    
    # Sky / Tunnel ceiling
    if not in_tunnel:
        draw_gradient_rect(draw, 0, 90, WIDTH, vp_y, (15, 25, 45), (35, 60, 95))
        # Distant horizon hills / skyline
        for i in range(12):
            bx = i * 180 - 40
            bw = 140
            bh = 60 + int(30 * math.sin(i * 1.5))
            draw.rectangle([(bx, vp_y - bh), (bx + bw, vp_y)], fill=(20, 35, 60))
    else:
        # Tunnel ceiling and overhead sodium/LED lamps
        draw_gradient_rect(draw, 0, 90, WIDTH, vp_y, (10, 10, 14), (25, 20, 15))
        # Tunnel arch lines
        for r in range(4):
            arc_y = vp_y - int(380 * (1 - r*0.2))
            draw.arc([(vp_x - 700 + r*50, arc_y), (vp_x + 700 - r*50, vp_y + 400)], start=180, end=0, fill=(45, 40, 35), width=3)
        # Overhead tunnel lights moving
        lamp_phase = (t * 4.0) % 1.0
        for l in range(6):
            lp = (lamp_phase + l) / 6.0
            ly = vp_y - int(lp**2 * 360)
            lx = vp_x
            lw = int(12 + lp * 60)
            lh = int(4 + lp * 12)
            glow_col = (255, 210, 120)
            draw.ellipse([(lx - lw//2, ly - lh//2), (lx + lw//2, ly + lh//2)], fill=glow_col)
    
    # Road surface
    road_col = (20, 24, 30) if in_tunnel else (30, 35, 42)
    draw.polygon([(vp_x - 30, vp_y), (vp_x + 30, vp_y), (WIDTH + 300, HEIGHT), (-300, HEIGHT)], fill=road_col)
    
    # Road side borders (yellow/white lines)
    draw.line([(vp_x - 30, vp_y), (-300, HEIGHT)], fill=(220, 220, 220), width=6)
    draw.line([(vp_x + 30, vp_y), (WIDTH + 300, HEIGHT)], fill=(220, 220, 220), width=6)
    
    # Moving dashed center road markings
    stripe_phase = (t * 2.5) % 1.0
    for s in range(8):
        p0 = (stripe_phase + s) / 8.0
        p1 = p0 + 0.06
        if p0 >= 1.0 or p1 >= 1.0:
            continue
        y0 = int(vp_y + (p0**2) * (HEIGHT - vp_y))
        y1 = int(vp_y + (p1**2) * (HEIGHT - vp_y))
        w0 = max(2, int(p0 * 18))
        w1 = max(3, int(p1 * 22))
        x0_l = vp_x - w0//2
        x0_r = vp_x + w0//2
        x1_l = vp_x - w1//2
        x1_r = vp_x + w1//2
        draw.polygon([(x0_l, y0), (x0_r, y0), (x1_r, y1), (x1_l, y1)], fill=(255, 255, 255))

def draw_car_interior(draw, phone_state, t):
    # Dark modern car dashboard at bottom
    dash_top = 740
    # Dashboard curve
    draw.polygon([(0, dash_top + 40), (450, dash_top), (WIDTH - 450, dash_top), (WIDTH, dash_top + 40),
                  (WIDTH, HEIGHT), (0, HEIGHT)], fill=(12, 15, 20))
    # Soft leather stitching / highlight
    draw.line([(0, dash_top + 40), (450, dash_top), (WIDTH - 450, dash_top), (WIDTH, dash_top + 40)],
              fill=(40, 50, 65), width=3)
    
    # Steering wheel rim visible on left
    draw.arc([(-150, 680), (350, 1180)], start=260, end=380, fill=(25, 30, 40), width=38)
    draw.arc([(-150, 680), (350, 1180)], start=260, end=380, fill=(50, 60, 75), width=4)
    
    # Dashboard gauge cluster (Speed & Gear)
    draw.rounded_rectangle([(180, dash_top + 30), (380, dash_top + 130)], radius=12, fill=(18, 24, 38))
    draw.text((200, dash_top + 42), "SPEED", font=f_badge, fill=C_TEXT_MUTED)
    draw.text((200, dash_top + 65), "65", font=f_large, fill=C_TEXT_WHITE)
    draw.text((285, dash_top + 95), "km/h", font=f_badge, fill=C_CYAN)

    # Smartphone mounted on center console / windshield mount
    # Phone coordinates
    px, py = 1240, 520
    pw, ph = 340, 520
    
    # Phone mount arm behind phone
    draw.rectangle([(px + pw//2 - 20, py + ph - 40), (px + pw//2 + 20, HEIGHT - 50)], fill=(25, 28, 35))
    draw.line([(px + pw//2 - 20, py + ph - 40), (px + pw//2 + 20, py + ph - 40)], fill=(60, 65, 75), width=2)
    
    # Phone outer case (dark titanium)
    draw.rounded_rectangle([(px, py), (px + pw, py + ph)], radius=28, fill=(20, 24, 32), outline=(75, 85, 99), width=3)
    
    # Phone screen bezel
    sx, sy = px + 12, py + 16
    sw, sh = pw - 24, ph - 32
    draw.rounded_rectangle([(sx, sy), (sx + sw, sy + sh)], radius=20, fill=(15, 23, 42))
    
    # Phone Speaker pill / Dynamic island
    draw.rounded_rectangle([(sx + sw//2 - 35, sy + 8), (sx + sw//2 + 35, sy + 22)], radius=7, fill=(8, 10, 14))
    
    # Phone UI Content based on phone_state
    draw_phone_screen_content(draw, sx, sy, sw, sh, phone_state, t)

def draw_phone_screen_content(draw, sx, sy, sw, sh, state, t):
    # State options: 'NORMAL_GNSS', 'GNSS_LOST', 'SUMARO_DR_ACTIVE', 'MAP_MATCHED', 'GNSS_RESTORED'
    map_y0 = sy + 36
    map_h = sh - 120
    
    # Draw stylized navigation map background
    draw.rectangle([(sx, map_y0), (sx + sw, map_y0 + map_h)], fill=(13, 20, 36))
    
    # Map grid / secondary roads
    for gx in range(sx + 20, sx + sw, 40):
        draw.line([(gx, map_y0), (gx, map_y0 + map_h)], fill=(20, 30, 50), width=1)
    for gy in range(map_y0 + 20, map_y0 + map_h, 40):
        draw.line([(sx, gy), (sx + sw, gy)], fill=(20, 30, 50), width=1)
        
    # Main highway route (curving through map)
    route_points = []
    for step in range(20):
        frac = step / 19.0
        rx = sx + sw//2 + int(35 * math.sin(frac * 3.5 + t * 0.5))
        ry = int((map_y0 + map_h) - frac * map_h)
        route_points.append((rx, ry))
    
    # Route corridor
    if len(route_points) >= 2:
        draw.line(route_points, fill=(30, 58, 138), width=18)
        draw.line(route_points, fill=(59, 130, 246), width=10)
        draw.line(route_points, fill=(147, 197, 253), width=3)
    
    # Current vehicle position arrow on phone map
    car_frac = 0.45
    cx = sx + sw//2 + int(35 * math.sin(car_frac * 3.5 + t * 0.5))
    cy = int((map_y0 + map_h) - car_frac * map_h)
    
    # Glitch or uncertainty offset if GNSS lost and not SUMARO
    if state == 'GNSS_LOST':
        # Jitter erratic jump
        cx += int(30 * math.sin(t * 18.0))
        cy += int(20 * math.cos(t * 14.0))
        # Red exploding uncertainty ring
        ring_r = int(25 + (t * 40.0) % 60)
        draw.ellipse([(cx - ring_r, cy - ring_r), (cx + ring_r, cy + ring_r)],
                     outline=C_RED, width=2)
    elif state in ['SUMARO_DR_ACTIVE', 'MAP_MATCHED']:
        # Smooth blue dead reckoning ring
        draw.ellipse([(cx - 16, cy - 16), (cx + 16, cy + 16)], outline=C_CYAN, width=2)
    
    # Arrow cursor
    arrow_col = C_GREEN if state in ['NORMAL_GNSS', 'GNSS_RESTORED'] else (C_RED if state == 'GNSS_LOST' else C_CYAN)
    draw.polygon([(cx, cy - 14), (cx - 10, cy + 10), (cx, cy + 5), (cx + 10, cy + 10)], fill=arrow_col)
    
    # Phone Top status bar
    draw.rectangle([(sx, sy + 30), (sx + sw, sy + 65)], fill=(17, 24, 39))
    if state in ['NORMAL_GNSS', 'GNSS_RESTORED']:
        draw.ellipse([(sx + 15, sy + 43), (sx + 25, sy + 53)], fill=C_GREEN)
        draw.text((sx + 34, sy + 38), "GNSS 3D FIX (10Hz)", font=f_badge, fill=C_GREEN)
    elif state == 'GNSS_LOST':
        draw.ellipse([(sx + 15, sy + 43), (sx + 25, sy + 53)], fill=C_RED)
        draw.text((sx + 34, sy + 38), "GNSS SIGNAL LOST", font=f_badge, fill=C_RED)
    elif state == 'SUMARO_DR_ACTIVE':
        draw.ellipse([(sx + 15, sy + 43), (sx + 25, sy + 53)], fill=C_CYAN)
        draw.text((sx + 34, sy + 38), "SUMARO DR ACTIVE", font=f_badge, fill=C_CYAN)
    elif state == 'MAP_MATCHED':
        draw.ellipse([(sx + 15, sy + 43), (sx + 25, sy + 53)], fill=C_PURPLE)
        draw.text((sx + 34, sy + 38), "MAP-MATCHED (99%)", font=f_badge, fill=C_PURPLE)

    # Phone Bottom Turn Card
    b_y = sy + sh - 95
    draw.rectangle([(sx, b_y), (sx + sw, sy + sh)], fill=(17, 24, 39))
    draw.text((sx + 15, b_y + 12), "CONTINUE ON TUNNEL EXPWY", font=f_badge, fill=C_TEXT_WHITE)
    draw.text((sx + 15, b_y + 40), "2.1 km  •  12 min  •  65 km/h", font=f_mono, fill=C_TEXT_MUTED)

# ==================== SCENE RENDERERS ====================

def render_scene_1(draw, frame_idx, t):
    # Scene 1: The Hook (0.0s - 8.0s)
    # Open highway, approaching tunnel entrance
    draw_road_perspective(draw, t, in_tunnel=False)
    
    # Approaching tunnel portal in distance (growing larger)
    tunnel_prog = min(1.0, t / 8.0)
    portal_w = int(240 + tunnel_prog * 450)
    portal_h = int(140 + tunnel_prog * 260)
    px0 = WIDTH // 2 - portal_w // 2
    py0 = 480 - portal_h + 30
    px1 = WIDTH // 2 + portal_w // 2
    py1 = 480 + 30
    
    # Concrete portal frame
    draw.rectangle([(px0 - 30, py0 - 40), (px1 + 30, py1)], fill=(55, 65, 81))
    draw.arc([(px0, py0 - 30), (px1, py1 + 40)], start=180, end=0, fill=(15, 20, 28), width=portal_h)
    draw.rectangle([(px0, py0 + 20), (px1, py1)], fill=(10, 12, 16))
    
    # Overhead Gantry Sign
    draw.rectangle([(px0 - 40, py0 - 80), (px1 + 40, py0 - 40)], fill=(30, 58, 138), outline=(255, 255, 255), width=2)
    draw.text((px0 + 20, py0 - 72), "CITY EXPRESSWAY TUNNEL • 2.5 KM", font=f_badge, fill=C_TEXT_WHITE)
    
    # Car cockpit with phone showing normal navigation
    draw_car_interior(draw, 'NORMAL_GNSS', t)
    
    # Scene header
    draw_header_banner(draw, "THE DEPENDENCE ON SATELLITE NAVIGATION", "SCENE 01 • THE HOOK", C_BLUE)
    
    # Lower cinematic caption
    cap = "We trust navigation to get us where we need to go..."
    w = draw.textlength(cap, font=f_body_bold)
    draw.rounded_rectangle([(WIDTH//2 - w//2 - 20, 100), (WIDTH//2 + w//2 + 20, 145)], radius=8, fill=(15, 23, 42, 220))
    draw.text((WIDTH//2 - w//2, 110), cap, font=f_body_bold, fill=C_TEXT_WHITE)

def render_scene_2(draw, frame_idx, t):
    # Scene 2: The Problem (8.0s - 18.0s)
    # Deep inside tunnel, GNSS drops out, navigation fails
    scene_t = t - 8.0
    draw_road_perspective(draw, t, in_tunnel=True)
    
    # Car cockpit with failed navigation
    phone_st = 'GNSS_LOST' if scene_t > 0.5 else 'NORMAL_GNSS'
    draw_car_interior(draw, phone_st, t)
    
    # Header
    draw_header_banner(draw, "SATELLITE OCCLUSION & NAVIGATION FAILURE", "SCENE 02 • THE PROBLEM", C_RED)
    
    # Outage Cause Info Cards on left side
    cards = [
        ("TUNNELS & UNDERPASSES", "Total satellite signal blockage", (220, 38, 38)),
        ("URBAN CANYONS", "Multipath & skyscraper occlusion", (234, 88, 12)),
        ("UNDERGROUND PARKING", "Zero line-of-sight to GNSS orbit", (185, 28, 28))
    ]
    for i, (title, desc, col) in enumerate(cards):
        cy = 130 + i * 110
        draw.rounded_rectangle([(60, cy), (500, cy + 90)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
        draw.rectangle([(60, cy + 10), (66, cy + 80)], fill=col)
        draw.text((85, cy + 18), title, font=f_body_bold, fill=C_TEXT_WHITE)
        draw.text((85, cy + 50), desc, font=f_body, fill=C_TEXT_MUTED)
        
    # Flashing warning overlay at center top
    if int(scene_t * 3) % 2 == 0:
        w_text = "⚠ STANDARD GPS CANNOT OPERATE WITHOUT SATELLITES"
        tw = draw.textlength(w_text, font=f_card_title)
        draw.rounded_rectangle([(WIDTH//2 - tw//2 - 25, 120), (WIDTH//2 + tw//2 + 25, 180)], radius=10, fill=(220, 38, 38))
        draw.text((WIDTH//2 - tw//2, 132), w_text, font=f_card_title, fill=(255, 255, 255))

def render_scene_3(draw, frame_idx, t):
    # Scene 3: The Question (18.0s - 22.0s)
    # Dark cinematic pause with glowing question typography
    scene_t = t - 18.0
    draw_gradient_rect(draw, 0, 0, WIDTH, HEIGHT, (7, 10, 19), (15, 23, 42))
    
    # Glowing ambient light grid
    for x in range(0, WIDTH, 80):
        draw.line([(x, 0), (x, HEIGHT)], fill=(20, 30, 50), width=1)
    for y in range(0, HEIGHT, 80):
        draw.line([(0, y), (WIDTH, y)], fill=(20, 30, 50), width=1)
        
    # Center question text
    q_line1 = "HOW DO WE KEEP NAVIGATING"
    q_line2 = "WITHOUT GNSS?"
    
    # Alpha pulse
    pulse = 0.8 + 0.2 * math.sin(scene_t * 4.0)
    col = (int(255 * pulse), int(255 * pulse), int(255 * pulse))
    
    w1 = draw.textlength(q_line1, font=f_large)
    w2 = draw.textlength(q_line2, font=f_large)
    
    draw.text((WIDTH//2 - w1//2, 380), q_line1, font=f_large, fill=col)
    draw.text((WIDTH//2 - w2//2, 470), q_line2, font=f_large, fill=C_CYAN)
    
    # Subtitle
    sub = "No Satellites  •  No Roadside Infrastructure  •  Zero Vehicle Modification"
    wsub = draw.textlength(sub, font=f_card_title)
    draw.text((WIDTH//2 - wsub//2, 590), sub, font=f_card_title, fill=C_TEXT_MUTED)
    
    # Glowing divider line
    line_w = int(min(1.0, scene_t / 1.5) * 600)
    draw.line([(WIDTH//2 - line_w, 560), (WIDTH//2 + line_w, 560)], fill=C_BLUE, width=3)

def render_scene_4(draw, frame_idx, t):
    # Scene 4: Introduce SUMARO (22.0s - 30.0s)
    # Phone breakdown & 4 internal sensors
    scene_t = t - 22.0
    draw_gradient_rect(draw, 0, 0, WIDTH, HEIGHT, (11, 16, 27), (17, 24, 39))
    draw_header_banner(draw, "THE SOLUTION: SMARTPHONE-BASED DEAD RECKONING", "SCENE 04 • INTRODUCING SUMARO", C_CYAN)
    
    # SUMARO Big Hero Logo Card on top
    draw.rounded_rectangle([(WIDTH//2 - 400, 110), (WIDTH//2 + 400, 210)], radius=16, fill=C_CARD_BG, outline=C_BLUE, width=2)
    logo_txt = "SUMARO"
    sub_txt = "AI/ML-Assisted Smartphone-Based Dead Reckoning System"
    lw = draw.textlength(logo_txt, font=f_large)
    sw = draw.textlength(sub_txt, font=f_body_bold)
    draw.text((WIDTH//2 - lw//2, 118), logo_txt, font=f_large, fill=C_CYAN)
    draw.text((WIDTH//2 - sw//2, 178), sub_txt, font=f_body_bold, fill=C_TEXT_WHITE)
    
    # 4 Sensor Breakdown Cards
    sensors = [
        ("GNSS RECEIVER", "BLOCKED / DENIED IN OUTAGES", "0 Hz fix rate during tunnel blackout", C_RED),
        ("3-AXIS ACCELEROMETER", "LINEAR ACCELERATION (Ax, Ay, Az)", "Captures vehicle forward & lateral thrust", C_GREEN),
        ("3-AXIS GYROSCOPE", "ANGULAR YAW RATE (ωx, ωy, ωz)", "Captures vehicle turns & heading changes", C_BLUE),
        ("3-AXIS MAGNETOMETER", "MAGNETIC HEADING VECTOR", "Absolute heading orientation reference", C_AMBER)
    ]
    
    for i, (name, role, detail, col) in enumerate(sensors):
        cx = 120 + (i % 2) * 860
        cy = 250 + (i // 2) * 220
        draw.rounded_rectangle([(cx, cy), (cx + 820, cy + 190)], radius=14, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
        # Accent bar
        draw.rectangle([(cx, cy + 15), (cx + 8, cy + 175)], fill=col)
        # Sensor title
        draw.text((cx + 30, cy + 25), name, font=f_card_title, fill=col)
        draw.text((cx + 30, cy + 75), role, font=f_body_bold, fill=C_TEXT_WHITE)
        draw.text((cx + 30, cy + 115), detail, font=f_body, fill=C_TEXT_MUTED)
        
        # Sensor status pill
        badge_txt = "ACTIVE" if col != C_RED else "DENIED"
        draw.rounded_rectangle([(cx + 680, cy + 25), (cx + 790, cy + 65)], radius=6, fill=col)
        bw = draw.textlength(badge_txt, font=f_badge)
        draw.text((cx + 735 - bw//2, cy + 34), badge_txt, font=f_badge, fill=(255, 255, 255))
        
    # Bottom callout banner
    draw.rounded_rectangle([(120, 720), (WIDTH - 120, 800)], radius=12, fill=(30, 58, 138))
    feat = "100% COTS SMARTPHONE • ZERO ADDITIONAL SENSORS • ZERO HARDWARE COST"
    fw = draw.textlength(feat, font=f_card_title)
    draw.text((WIDTH//2 - fw//2, 742), feat, font=f_card_title, fill=(255, 255, 255))

def render_scene_5(draw, frame_idx, t):
    # Scene 5: Calibration + Normal Navigation (30.0s - 39.0s)
    scene_t = t - 30.0
    draw_gradient_rect(draw, 0, 0, WIDTH, HEIGHT, (11, 16, 27), (17, 24, 39))
    draw_header_banner(draw, "AUTO-CALIBRATION & NOMINAL SENSOR FUSION", "SCENE 05 • CALIBRATION", C_GREEN)
    
    # Left Panel: Phone Mount & Tilt Angles
    draw.rounded_rectangle([(80, 120), (900, 780)], radius=16, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((110, 145), "PHONE-TO-VEHICLE DYNAMIC ALIGNMENT", font=f_card_title, fill=C_TEXT_WHITE)
    draw.text((110, 185), "The phone can be mounted at any arbitrary dashboard angle.", font=f_body, fill=C_TEXT_MUTED)
    
    # Diagram of Phone vs Vehicle Coordinate Frame
    center_x, center_y = 480, 430
    # Vehicle axes (cyan)
    draw.line([(center_x, center_y), (center_x + 220, center_y)], fill=C_CYAN, width=4)
    draw.text((center_x + 230, center_y - 12), "Vehicle X (Forward)", font=f_body_bold, fill=C_CYAN)
    draw.line([(center_x, center_y), (center_x, center_y - 200)], fill=C_CYAN, width=4)
    draw.text((center_x - 70, center_y - 230), "Vehicle Z (Up)", font=f_body_bold, fill=C_CYAN)
    
    # Phone tilted frame (orange, 30 deg pitch)
    pitch_rad = math.radians(30)
    px_end = center_x + int(200 * math.cos(-pitch_rad))
    py_end = center_y + int(200 * math.sin(-pitch_rad))
    draw.line([(center_x, center_y), (px_end, py_end)], fill=C_AMBER, width=4)
    draw.text((px_end + 15, py_end - 10), "Phone X' (Tilted 30°)", font=f_body_bold, fill=C_AMBER)
    
    # Tilt arc
    draw.arc([(center_x - 80, center_y - 80), (center_x + 80, center_y + 80)], start=-30, end=0, fill=C_TEXT_WHITE, width=2)
    draw.text((center_x + 95, center_y - 25), "θ = 29.8°", font=f_mono, fill=C_TEXT_WHITE)
    
    # Telemetry badges at bottom of left panel
    draw.rounded_rectangle([(110, 660), (870, 740)], radius=10, fill=(15, 23, 42))
    calib_st = "PITCH: 29.8°  |  ROLL: 1.2°  |  DCM MATRIX: CONVERGED  |  COV < 10⁻⁴"
    draw.text((130, 688), calib_st, font=f_mono, fill=C_GREEN)
    
    # Right Panel: Normal State Sensor Fusion Data-Flow
    draw.rounded_rectangle([(940, 120), (WIDTH - 80, 780)], radius=16, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((970, 145), "NOMINAL EKF SENSOR FUSION (GNSS ON)", font=f_card_title, fill=C_TEXT_WHITE)
    
    blocks = [
        ("10 Hz SMARTPHONE IMU", "Accelerations & Yaw Rates preprocessed", C_BLUE),
        ("ORIENTATION ESTIMATOR", "Leveling & gravity vector subtraction", C_AMBER),
        ("1 Hz GNSS FIX", "Latitude, longitude, speed & heading updates", C_GREEN),
        ("6-STATE KALMAN FILTER", "Continuous position [x, y, vx, vy, θ, bg] estimation", C_PURPLE)
    ]
    for b_idx, (b_title, b_sub, b_col) in enumerate(blocks):
        by = 220 + b_idx * 130
        draw.rounded_rectangle([(970, by), (WIDTH - 110, by + 100)], radius=12, fill=(15, 23, 42), outline=b_col, width=2)
        draw.text((1000, by + 18), b_title, font=f_body_bold, fill=b_col)
        draw.text((1000, by + 54), b_sub, font=f_body, fill=C_TEXT_MUTED)
        # Downward connector arrow
        if b_idx < 3:
            draw.line([(1400, by + 100), (1400, by + 130)], fill=C_TEXT_MUTED, width=3)
            draw.polygon([(1395, by + 125), (1405, by + 125), (1400, by + 130)], fill=C_TEXT_WHITE)

def render_scene_6(draw, frame_idx, t):
    # Scene 6: GNSS Outage / SUMARO Activates (39.0s - 48.0s)
    # Cockpit in tunnel with hero SUMARO activation banner & live waveforms
    scene_t = t - 39.0
    draw_road_perspective(draw, t, in_tunnel=True)
    draw_car_interior(draw, 'SUMARO_DR_ACTIVE', t)
    
    # Header
    draw_header_banner(draw, "SEAMLESS TRANSITION TO DEAD RECKONING", "SCENE 06 • SUMARO ACTIVATES", C_CYAN)
    
    # Hero Floating HUD Banner across top center
    banner_w = 920
    bx0 = WIDTH // 2 - banner_w // 2 - 120
    draw.rounded_rectangle([(bx0, 110), (bx0 + banner_w, 200)], radius=14, fill=(15, 23, 42), outline=C_CYAN, width=3)
    draw.text((bx0 + 30, 122), "⚡ SUMARO DEAD RECKONING ENGINE ACTIVE", font=f_card_title, fill=C_CYAN)
    draw.text((bx0 + 30, 162), "GNSS DENIED  •  ESTIMATING FROM 100% INERTIAL PHONE DYNAMICS", font=f_body_bold, fill=C_TEXT_WHITE)
    
    # Left HUD Card: Live Waveforms
    draw.rounded_rectangle([(60, 230), (580, 710)], radius=14, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((85, 250), "REAL-TIME INERTIAL STREAM", font=f_card_title, fill=C_TEXT_WHITE)
    
    # Waveform 1: Forward Acceleration (m/s²)
    draw.text((85, 300), "LONGITUDINAL ACCEL (a_fwd)", font=f_badge, fill=C_CYAN)
    draw.rectangle([(85, 330), (555, 470)], fill=(12, 16, 25), outline=(30, 41, 59), width=1)
    # Zero line
    draw.line([(85, 400), (555, 400)], fill=(40, 50, 70), width=1)
    pts_a = []
    for step in range(70):
        fx = 85 + int(step * 6.7)
        fy = 400 - int(25 * math.sin(step * 0.25 - scene_t * 6.0) + 10 * math.cos(step * 0.7))
        pts_a.append((fx, fy))
    draw.line(pts_a, fill=C_CYAN, width=2)
    
    # Waveform 2: Gyro Yaw Rate (deg/s)
    draw.text((85, 490), "ANGULAR YAW RATE (ω_z)", font=f_badge, fill=C_AMBER)
    draw.rectangle([(85, 520), (555, 660)], fill=(12, 16, 25), outline=(30, 41, 59), width=1)
    draw.line([(85, 590), (555, 590)], fill=(40, 50, 70), width=1)
    pts_g = []
    for step in range(70):
        fx = 85 + int(step * 6.7)
        fy = 590 - int(30 * math.sin(step * 0.18 - scene_t * 4.0))
        pts_g.append((fx, fy))
    draw.line(pts_g, fill=C_AMBER, width=2)
    
    # Key Callout at bottom left
    draw.text((85, 675), "Continuous 50 Hz motion tracking active", font=f_mono, fill=C_GREEN)

def render_scene_7(draw, frame_idx, t):
    # Scene 7: Drift + AI/ML (48.0s - 59.0s)
    # Top-down schematic showing quadratic drift vs ML residual correction
    scene_t = t - 48.0
    draw_gradient_rect(draw, 0, 0, WIDTH, HEIGHT, (11, 16, 27), (17, 24, 39))
    draw_header_banner(draw, "THE DRIFT CHALLENGE & AI/ML RESIDUAL CORRECTION", "SCENE 07 • SENSOR DRIFT & ML", C_AMBER)
    
    # Left Section: Trajectory Comparison on Highway Grid
    draw.rounded_rectangle([(60, 110), (940, 790)], radius=16, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((90, 135), "TRAJECTORY DIVERGENCE DURING OUTAGE", font=f_card_title, fill=C_TEXT_WHITE)
    
    # Map area inside left section
    mx0, my0, mx1, my1 = 90, 185, 910, 680
    draw.rectangle([(mx0, my0), (mx1, my1)], fill=(12, 18, 30))
    # Grid lines
    for gx in range(mx0, mx1, 60):
        draw.line([(gx, my0), (gx, my1)], fill=(20, 30, 48), width=1)
    for gy in range(my0, my1, 60):
        draw.line([(mx0, gy), (mx1, gy)], fill=(20, 30, 48), width=1)
        
    # Road curve (True Path - Green)
    true_pts = [(mx0 + 40, my1 - 60), (mx0 + 250, my1 - 120), (mx0 + 480, my1 - 250), (mx0 + 680, my1 - 380), (mx1 - 40, my1 - 440)]
    draw.line(true_pts, fill=C_GREEN, width=12)
    draw.text((mx1 - 320, my1 - 475), "TRUE ROAD CENTERLINE", font=f_badge, fill=C_GREEN)
    
    # Unassisted IMU Double Integration (Red dashed drifting away into wall)
    drift_pts = [(mx0 + 40, my1 - 60), (mx0 + 250, my1 - 100), (mx0 + 480, my1 - 140), (mx0 + 680, my1 - 160), (mx1 - 40, my1 - 170)]
    draw.line(drift_pts, fill=C_RED, width=4)
    draw.text((mx1 - 380, my1 - 200), "RAW IMU DRIFT (Quadratic Error)", font=f_badge, fill=C_RED)
    
    # SUMARO ML-Corrected Path (Cyan - closely hugging true path)
    sumaro_pts = [(mx0 + 40, my1 - 60), (mx0 + 250, my1 - 118), (mx0 + 480, my1 - 242), (mx0 + 680, my1 - 370), (mx1 - 40, my1 - 425)]
    draw.line(sumaro_pts, fill=C_CYAN, width=6)
    draw.text((mx1 - 360, my1 - 410), "SUMARO AI/ML CORRECTED", font=f_badge, fill=C_CYAN)
    
    # Left stats pill
    draw.rounded_rectangle([(90, 705), (910, 765)], radius=8, fill=(15, 23, 42))
    draw.text((115, 725), "HELD-OUT VW1 BENCHMARK: Max Error 9,512m (EKF) → 805m (SUMARO RBPF+ML)", font=f_mono, fill=C_CYAN)
    
    # Right Section: AI/ML Motion Pattern & Residual Pipeline
    draw.rounded_rectangle([(980, 110), (WIDTH - 60, 790)], radius=16, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((1010, 135), "CAUSAL ML RESIDUAL ARCHITECTURE", font=f_card_title, fill=C_TEXT_WHITE)
    
    ml_steps = [
        ("1. CAUSAL FEATURE EXTRACTION (W = 50)", "Rolling 5-second statistical window (mean, std, jerk, stationary score)", C_BLUE),
        ("2. HIST GRADIENT BOOSTED REGRESSOR", "Learns vehicle dynamics & vibration patterns without future leakage", C_AMBER),
        ("3. FORWARD ACCEL RESIDUAL (δa_fwd)", "Directly removes tilt-induced gravity & sensor bias offset", C_GREEN),
        ("4. GYRO YAW RATE RESIDUAL (δω_z)", "Stabilizes unobserved heading drift before Kalman integration", C_CYAN)
    ]
    for idx, (head, det, col) in enumerate(ml_steps):
        sy = 195 + idx * 140
        draw.rounded_rectangle([(1010, sy), (WIDTH - 90, sy + 115)], radius=12, fill=(15, 23, 42), outline=col, width=2)
        draw.text((1035, sy + 20), head, font=f_body_bold, fill=col)
        draw.text((1035, sy + 58), det, font=f_body, fill=C_TEXT_MUTED)

def render_scene_8(draw, frame_idx, t):
    # Scene 8: Sensor Fusion + Offline Map Matching (59.0s - 70.0s)
    # Showing EKF filter & vector map matching snapping to road
    scene_t = t - 59.0
    draw_gradient_rect(draw, 0, 0, WIDTH, HEIGHT, (11, 16, 27), (17, 24, 39))
    draw_header_banner(draw, "ADAPTIVE SENSOR FUSION & VECTOR MAP MATCHING", "SCENE 08 • FUSION & MAP MATCHING", C_PURPLE)
    
    # Left Section: Kalman Filter / Particle Filter State Engine
    draw.rounded_rectangle([(60, 110), (920, 790)], radius=16, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((90, 135), "ADAPTIVE ESTIMATOR (EKF / RBPF)", font=f_card_title, fill=C_TEXT_WHITE)
    
    # State Vector Representation
    draw.rounded_rectangle([(90, 190), (890, 310)], radius=12, fill=(15, 23, 42), outline=C_BLUE, width=2)
    draw.text((115, 210), "STATE VECTOR x_t = [ px,  py,  vx,  vy,  θ,  b_gyro ]ᵀ", font=f_card_title, fill=C_CYAN)
    draw.text((115, 260), "px, py (2D Pos)  •  vx, vy (Velocity)  •  θ (Heading)  •  b_gyro (Sensor Bias)", font=f_body, fill=C_TEXT_MUTED)
    
    # Mathematical Guarantee Cards
    feats = [
        ("JOSEPH-FORM COVARIANCE", "Maintains positive semi-definiteness and numerical stability across hours.", C_GREEN),
        ("INNOVATION GATING (NIS)", "Rejects anomalous spurious spikes via Mahalanobis distance chi-squared gating.", C_AMBER),
        ("DYNAMIC PROCESS NOISE Q", "Adaptive Q-scaling automatically expands uncertainty during high maneuvers.", C_PURPLE)
    ]
    for i, (head, desc, col) in enumerate(feats):
        fy = 330 + i * 140
        draw.rounded_rectangle([(90, fy), (890, fy + 115)], radius=10, fill=(15, 23, 42), outline=C_CARD_BORDER, width=1)
        draw.rectangle([(90, fy + 10), (96, fy + 105)], fill=col)
        draw.text((120, fy + 20), head, font=f_body_bold, fill=col)
        draw.text((120, fy + 58), desc, font=f_body, fill=C_TEXT_MUTED)
        
    # Right Section: Offline Vector Map Matching Engine
    draw.rounded_rectangle([(960, 110), (WIDTH - 60, 790)], radius=16, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((990, 135), "OFFLINE TOPOLOGICAL MAP MATCHING", font=f_card_title, fill=C_TEXT_WHITE)
    
    # Map Graph Visualizer
    gx0, gy0, gx1, gy1 = 990, 190, WIDTH - 90, 640
    draw.rectangle([(gx0, gy0), (gx1, gy1)], fill=(12, 18, 30))
    
    # Road network links
    nodes = [(gx0 + 60, gy1 - 50), (gx0 + 260, gy1 - 120), (gx0 + 520, gy1 - 250), (gx1 - 60, gy1 - 380)]
    draw.line(nodes, fill=(40, 60, 95), width=24)
    draw.line(nodes, fill=C_PURPLE, width=8)
    for nx, ny in nodes:
        draw.ellipse([(nx - 10, ny - 10), (nx + 10, ny + 10)], fill=C_TEXT_WHITE)
    
    # Candidate Projection Arcs from Off-Road DR to Road Centerline
    snap_t = min(1.0, max(0.0, (scene_t - 5.0) / 2.0))
    dr_x = gx0 + 360
    dr_y = gy1 - 120  # off road
    road_x = gx0 + 380
    road_y = gy1 - 180 # on road
    
    # Interpolate cursor position based on snap
    curr_x = int(dr_x * (1 - snap_t) + road_x * snap_t)
    curr_y = int(dr_y * (1 - snap_t) + road_y * snap_t)
    
    # Dashed snap projection line
    draw.line([(dr_x, dr_y), (road_x, road_y)], fill=C_AMBER, width=2)
    draw.ellipse([(curr_x - 14, curr_y - 14), (curr_x + 14, curr_y + 14)], fill=C_CYAN, outline=(255, 255, 255), width=3)
    
    # Map Match Status Tag
    if snap_t >= 0.9:
        draw.rounded_rectangle([(gx0 + 20, gy0 + 20), (gx0 + 440, gy0 + 80)], radius=8, fill=C_PURPLE)
        draw.text((gx0 + 40, gy0 + 34), "✓ ROAD LINK #4812 SNAPPED (99.4%)", font=f_badge, fill=(255, 255, 255))
    else:
        draw.rounded_rectangle([(gx0 + 20, gy0 + 20), (gx0 + 440, gy0 + 80)], radius=8, fill=(30, 41, 59))
        draw.text((gx0 + 40, gy0 + 34), "SEARCHING CANDIDATE ROAD LINKS...", font=f_badge, fill=C_TEXT_MUTED)

    # Bottom explanation text
    draw.text((990, 670), "Uses OpenStreetMap vector topology stored 100% offline on-device.", font=f_body, fill=C_TEXT_MUTED)
    draw.text((990, 710), "Ensures vehicle never drifts across walls or invalid terrain.", font=f_body_bold, fill=C_TEXT_WHITE)

def render_scene_9(draw, frame_idx, t):
    # Scene 9: GNSS Returns (70.0s - 80.0s)
    # Emerging from tunnel into bright daylight, GNSS restored smoothly
    scene_t = t - 70.0
    daylight_frac = min(1.0, scene_t / 3.0)
    
    # Render road exiting tunnel into daylight
    draw_road_perspective(draw, t, in_tunnel=(daylight_frac < 0.5))
    
    # Cockpit view with phone showing GNSS restored
    draw_car_interior(draw, 'GNSS_RESTORED', t)
    draw_header_banner(draw, "RECOVERY & SEAMLESS STATE RECONCILIATION", "SCENE 09 • GNSS RESTORED", C_GREEN)
    
    # Center Success Banner
    banner_w = 900
    bx0 = WIDTH // 2 - banner_w // 2 - 120
    draw.rounded_rectangle([(bx0, 110), (bx0 + banner_w, 200)], radius=14, fill=(15, 23, 42), outline=C_GREEN, width=3)
    draw.text((bx0 + 35, 122), "🟢 GNSS SATELLITE FIX RESTORED", font=f_card_title, fill=C_GREEN)
    draw.text((bx0 + 35, 162), "12 SATELLITES ACQUIRED  •  SMOOTH KALMAN INNOVATION RECONCILIATION", font=f_body_bold, fill=C_TEXT_WHITE)
    
    # Left Comparison Card: Smooth vs Jumping
    draw.rounded_rectangle([(60, 240), (580, 710)], radius=14, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((85, 265), "ZERO-TELEPORT RECONCILIATION", font=f_card_title, fill=C_TEXT_WHITE)
    
    points = [
        ("CONVENTIONAL GPS", "Teleports violently across map when fix returns, confusing driver.", C_RED),
        ("SUMARO DEAD RECKONING", "Continuous estimate is already within meters of true GNSS fix.", C_CYAN),
        ("SMOOTH COVARIANCE BLEND", "Kalman innovation absorbs residual without sudden jumping.", C_GREEN)
    ]
    for p_idx, (p_title, p_desc, p_col) in enumerate(points):
        py = 330 + p_idx * 120
        draw.rounded_rectangle([(85, py), (555, py + 95)], radius=10, fill=(15, 23, 42))
        draw.rectangle([(85, py + 8), (91, py + 87)], fill=p_col)
        draw.text((105, py + 16), p_title, font=f_body_bold, fill=p_col)
        draw.text((105, py + 48), p_desc, font=f_body, fill=C_TEXT_MUTED)

def render_scene_10(draw, frame_idx, t):
    # Scene 10: Driver Experience (80.0s - 84.7s)
    # Split triptych showing the 3 seamless driver states
    draw_gradient_rect(draw, 0, 0, WIDTH, HEIGHT, (11, 16, 27), (17, 24, 39))
    draw_header_banner(draw, "THE DRIVER PERSPECTIVE: ZERO DISTRACTION", "SCENE 10 • DRIVER EXPERIENCE", C_BLUE)
    
    states = [
        ("1. NORMAL DRIVING", "🟢 GNSS ACTIVE", "Turn-by-turn navigation running normally on satellite signals.", C_GREEN),
        ("2. TUNNEL / OUTAGE", "🔴 SUMARO DR ACTIVE", "Automatic dead-reckoning switch. Screen continues smoothly.", C_CYAN),
        ("3. GNSS RESTORATION", "🟢 GNSS RESTORED", "Seamless return. Zero map jumping. Zero driver confusion.", C_GREEN)
    ]
    
    card_w = 540
    card_h = 580
    for idx, (st_title, st_badge, st_desc, st_col) in enumerate(states):
        cx = 80 + idx * 600
        cy = 130
        draw.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=16, fill=C_CARD_BG, outline=st_col, width=2)
        # Header inside card
        draw.text((cx + 30, cy + 30), st_title, font=f_card_title, fill=C_TEXT_WHITE)
        # Status Pill
        draw.rounded_rectangle([(cx + 30, cy + 80), (cx + card_w - 30, cy + 130)], radius=8, fill=(15, 23, 42))
        draw.text((cx + 50, cy + 92), st_badge, font=f_body_bold, fill=st_col)
        
        # Mock mini map inside card
        map_y = cy + 150
        map_h = 280
        draw.rounded_rectangle([(cx + 30, map_y), (cx + card_w - 30, map_y + map_h)], radius=10, fill=(13, 20, 36))
        # Draw road ribbon
        draw.line([(cx + 70, map_y + 240), (cx + card_w//2, map_y + 140), (cx + card_w - 70, map_y + 40)], fill=C_BLUE, width=12)
        # Vehicle marker
        draw.ellipse([(cx + card_w//2 - 12, map_y + 140 - 12), (cx + card_w//2 + 12, map_y + 140 + 12)], fill=st_col, outline=(255, 255, 255), width=2)
        
        # Description text below map
        draw.text((cx + 30, cy + 460), st_desc, font=f_body, fill=C_TEXT_MUTED)
        
    # Bottom Summary Tag
    draw.rounded_rectangle([(80, 740), (WIDTH - 80, 810)], radius=12, fill=(30, 58, 138))
    sum_t = "FOR THE DRIVER, IT STAYS SIMPLE: JUST NAVIGATION THAT KEEPS WORKING."
    sw = draw.textlength(sum_t, font=f_card_title)
    draw.text((WIDTH//2 - sw//2, 758), sum_t, font=f_card_title, fill=(255, 255, 255))

def render_scene_11(draw, frame_idx, t):
    # Scene 11: Final USP & End Card (84.7s - 90.0s)
    # Master cinematic presentation card
    scene_t = t - 84.7
    fade_end = max(0.0, min(1.0, (90.0 - t) / 0.5))  # Fade to black in last 0.5s
    
    draw_gradient_rect(draw, 0, 0, WIDTH, HEIGHT, (7, 10, 18), (15, 23, 42))
    
    # Elegant glowing tech geometric lines in background
    for r in range(1, 6):
        draw.circle((WIDTH // 2, 440), r * 140, outline=(20, 35, 60), width=1)
    
    # Large SUMARO Typography
    logo_txt = "SUMARO"
    lw = draw.textlength(logo_txt, font=f_large)
    draw.text((WIDTH//2 - lw//2, 220), logo_txt, font=f_large, fill=C_CYAN)
    
    sub = "AI/ML-Assisted Smartphone-Based Dead Reckoning"
    sw = draw.textlength(sub, font=f_card_title)
    draw.text((WIDTH//2 - sw//2, 310), sub, font=f_card_title, fill=C_TEXT_WHITE)
    
    # Accent line
    draw.line([(WIDTH//2 - 400, 375), (WIDTH//2 + 400, 375)], fill=C_BLUE, width=3)
    
    # Main Tagline
    tagline_1 = "WHEN GNSS DISAPPEARS,"
    tagline_2 = "SUMARO DOESN'T STOP NAVIGATING."
    t1_w = draw.textlength(tagline_1, font=f_title)
    t2_w = draw.textlength(tagline_2, font=f_title)
    draw.text((WIDTH//2 - t1_w//2, 420), tagline_1, font=f_title, fill=C_TEXT_WHITE)
    draw.text((WIDTH//2 - t2_w//2, 495), tagline_2, font=f_title, fill=C_CYAN)
    
    # Core Three Pillars
    p1 = "IT ESTIMATES."
    p2 = "IT CORRECTS."
    p3 = "IT KEEPS GOING."
    pw1 = draw.textlength(p1, font=f_card_title)
    pw2 = draw.textlength(p2, font=f_card_title)
    pw3 = draw.textlength(p3, font=f_card_title)
    
    total_pw = pw1 + pw2 + pw3 + 120
    px = WIDTH // 2 - total_pw // 2
    draw.text((px, 600), p1, font=f_card_title, fill=C_TEXT_MUTED)
    draw.text((px + pw1 + 60, 600), p2, font=f_card_title, fill=C_AMBER)
    draw.text((px + pw1 + pw2 + 120, 600), p3, font=f_card_title, fill=C_GREEN)
    
    # Smart India Hackathon 2026 Footer Badge
    draw.rounded_rectangle([(WIDTH//2 - 320, 720), (WIDTH//2 + 320, 780)], radius=30, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    sih = "SMART INDIA HACKATHON 2026 • SIH 2026"
    sih_w = draw.textlength(sih, font=f_badge)
    draw.text((WIDTH//2 - sih_w//2, 742), sih, font=f_badge, fill=C_CYAN)
    
    # Fade to black overlay if near end
    if fade_end < 1.0:
        black_img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
        # Draw semi-transparent black
        alpha = int(255 * (1.0 - fade_end))
        draw.rectangle([(0, 0), (WIDTH, HEIGHT)], fill=(0, 0, 0, alpha))

def generate_video():
    print(f"==================================================")
    print(f"SUMARO 90-SECOND SIH 2026 SOLUTION VIDEO GENERATOR")
    print(f"==================================================")
    print(f"Resolution: {WIDTH}x{HEIGHT} | FPS: {FPS} | Total Frames: {TOTAL_FRAMES} (90.0s)")
    print(f"Audio Track: {AUDIO_MIX}")
    print(f"Output Video: {OUTPUT_VIDEO}")
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # FFmpeg command piping raw BGR video frames and merging final audio
    cmd = [
        ffmpeg_exe,
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-pix_fmt', 'bgr24',
        '-r', str(FPS),
        '-i', '-',                # Pipe raw video frames from stdin
        '-i', AUDIO_MIX,          # Master Audio Track
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '19',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        OUTPUT_VIDEO
    ]
    
    print("Launching FFmpeg encoder process...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    img = Image.new('RGB', (WIDTH, HEIGHT), C_BG_DARK)
    draw = ImageDraw.Draw(img)
    
    print("Rendering 2,700 frames...")
    
    for f in range(TOTAL_FRAMES):
        t = f / float(FPS)
        
        # Reset canvas
        draw.rectangle([(0, 0), (WIDTH, HEIGHT)], fill=C_BG_DARK)
        
        # Route to appropriate scene
        if t < 8.0:
            render_scene_1(draw, f, t)
        elif t < 18.0:
            render_scene_2(draw, f, t)
        elif t < 22.0:
            render_scene_3(draw, f, t)
        elif t < 30.0:
            render_scene_4(draw, f, t)
        elif t < 39.0:
            render_scene_5(draw, f, t)
        elif t < 48.0:
            render_scene_6(draw, f, t)
        elif t < 59.0:
            render_scene_7(draw, f, t)
        elif t < 70.0:
            render_scene_8(draw, f, t)
        elif t < 80.0:
            render_scene_9(draw, f, t)
        elif t < 84.7:
            render_scene_10(draw, f, t)
        else:
            render_scene_11(draw, f, t)
            
        # Convert RGB PIL Image to BGR raw bytes for FFmpeg
        arr = np.array(img)
        bgr = arr[:, :, ::-1]  # RGB to BGR
        proc.stdin.write(bgr.tobytes())
        
        if (f + 1) % 300 == 0 or f == TOTAL_FRAMES - 1:
            pct = ((f + 1) / TOTAL_FRAMES) * 100.0
            print(f"Rendered frame {f+1:4d} / {TOTAL_FRAMES} ({pct:5.1f}%) | Time: {t:4.1f}s / 90.0s")
            
    print("Closing stdin and waiting for FFmpeg to finalize encoding...")
    proc.stdin.close()
    proc.wait()
    
    # Check if file was created successfully
    if os.path.exists(OUTPUT_VIDEO):
        vsize_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
        print(f"SUCCESS: Video rendered successfully!")
        print(f"File: {OUTPUT_VIDEO} ({vsize_mb:.2f} MB)")
        
        # Copy to root as well
        import shutil
        shutil.copy2(OUTPUT_VIDEO, ROOT_COPY)
        print(f"Copied video to workspace root: {ROOT_COPY}")
    else:
        print(f"ERROR: Video file was not created!")
        stderr_output = proc.stderr.read().decode('utf-8', errors='ignore')
        print("FFmpeg Stderr:\n", stderr_output)

if __name__ == "__main__":
    generate_video()
