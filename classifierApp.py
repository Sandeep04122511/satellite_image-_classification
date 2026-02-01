# classifierApp.py 
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import logging
from PIL import Image, ImageTk, ImageOps, ExifTags
import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
import webbrowser
import tempfile
import io
import math
import os
from datetime import datetime
import requests
import base64

# Import the img_to_array function from Keras
try:
    from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
except ImportError:
    # Fallback implementation if Keras is not available
    def img_to_array(img, data_format=None):
        """Converts a PIL Image instance to a Numpy array."""
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = np.stack((img_array,) * 3, axis=-1)
        return img_array

from classifier import SatelliteImageClassifier, AuthenticationError, RegistrationError, ModelLoadError, ImageProcessingError

# Configure customtkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_exif_location(image_path):
    """Extract GPS coordinates from image EXIF data if available"""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if not exif_data:
            return None, None
            
        # Get GPS tags
        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                gps_info = value
                break
                
        if not gps_info:
            return None, None
            
        # Extract latitude and longitude
        def get_value(tag_id, default=None):
            return gps_info.get(tag_id, default)
            
        # Process latitude
        lat_deg = get_value(1, [0, 1])
        lat_min = get_value(2, [0, 1])
        lat_sec = get_value(3, [0, 1])
        lat_ref = get_value(4, 'N')
        
        # Process longitude
        lon_deg = get_value(5, [0, 1])
        lon_min = get_value(6, [0, 1])
        lon_sec = get_value(7, [0, 1])
        lon_ref = get_value(8, 'E')
        
        # Convert to decimal degrees
        lat = lat_deg[0] / lat_deg[1] + lat_min[0] / (lat_min[1] * 60) + lat_sec[0] / (lat_sec[1] * 3600)
        lon = lon_deg[0] / lon_deg[1] + lon_min[0] / (lon_min[1] * 60) + lon_sec[0] / (lon_sec[1] * 3600)
        
        # Apply direction
        if lat_ref == 'S':
            lat = -lat
        if lon_ref == 'W':
            lon = -lon
            
        return lat, lon
        
    except Exception as e:
        logger.warning(f"Error extracting EXIF location: {str(e)}")
        return None, None

class ModernHomePage(ctk.CTkFrame):
    """Modern Home Page with Satellite Background"""
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        # Main container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=40, pady=40)

        # Header with logo
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(pady=(50, 20))

        # Logo/Title
        title_label = ctk.CTkLabel(header_frame, 
                                  text="🛰️ Satellite Image Classification System",
                                  font=("Arial", 32, "bold"),
                                  text_color="#4FC3F7")
        title_label.pack(pady=10)

        subtitle_label = ctk.CTkLabel(header_frame,
                                     text="Advanced AI-Powered satellite Image Classification",
                                     font=("Arial", 16),
                                     text_color="#B3E5FC")
        subtitle_label.pack()

        # Features grid
        features_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        features_frame.pack(pady=50, fill="both", expand=True)

        features = [
            {"icon": "🌿", "title": "Land Cover Classification", "desc": "Identify vegetation, water, urban areas"},
            {"icon": "📊", "title": "Grid-Based Analysis", "desc": "Detailed segment-by-segment analysis"},
            {"icon": "🔒", "title": "Secure Authentication", "desc": "Protected user accounts and data"},
            {"icon": "🌍", "title": "Global Satellite Imagery", "desc": "Esri World Imagery for worldwide access"},
            {"icon": "🖼️", "title": "Multi-Format Support", "desc": "JPG, PNG, TIFF image formats"},
            {"icon": "📋", "title": "Comprehensive Reports", "desc": "Export KML files and detailed analysis"}
        ]

        for i, feature in enumerate(features):
            row = i // 3
            col = i % 3
            feature_card = self.create_feature_card(features_frame, feature)
            feature_card.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")
            features_frame.grid_columnconfigure(col, weight=1)
            features_frame.grid_rowconfigure(row, weight=1)

        # Action buttons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=40)

        login_btn = ctk.CTkButton(button_frame,
                                 text="🚀 Get Started - Login",
                                 command=self.app.show_login,
                                 font=("Arial", 16, "bold"),
                                 height=50,
                                 width=200,
                                 corner_radius=25,
                                 fg_color="#1976D2",
                                 hover_color="#1565C0")
        login_btn.pack(side="left", padx=15)

        register_btn = ctk.CTkButton(button_frame,
                                    text="📝 Create Account",
                                    command=self.app.show_register,
                                    font=("Arial", 16, "bold"),
                                    height=50,
                                    width=200,
                                    corner_radius=25,
                                    fg_color="#388E3C",
                                    hover_color="#2E7D32")
        register_btn.pack(side="left", padx=15)
        
        # Contact button
        contact_btn = ctk.CTkButton(button_frame,
                                   text="👥 Contact Us",
                                   command=self.app.show_contact,
                                   font=("Arial", 16, "bold"),
                                   height=50,
                                   width=200,
                                   corner_radius=25,
                                   fg_color="#9C27B0",
                                   hover_color="#7B1FA2")
        contact_btn.pack(side="left", padx=15)

        # Footer
        footer_label = ctk.CTkLabel(main_frame,
                                   text=" Satellite Image Classification System • Advanced Geospatial Analysis",
                                   font=("Arial", 12),
                                   text_color="#90A4AE")
        footer_label.pack(side="bottom", pady=20)

    def create_feature_card(self, parent, feature):
        card = ctk.CTkFrame(parent, 
                           corner_radius=20,
                           fg_color=("#E3F2FD", "#1E2A38"),
                           border_width=2,
                           border_color=("#BBDEFB", "#37474F"))
        card.pack_propagate(False)
        card.configure(width=300, height=150)

        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(content_frame, 
                    text=feature["icon"],
                    font=("Arial", 24)).pack(anchor="w")
        
        ctk.CTkLabel(content_frame,
                    text=feature["title"],
                    font=("Arial", 16, "bold"),
                    text_color=("#1565C0", "#4FC3F7")).pack(anchor="w", pady=(10, 5))
        
        ctk.CTkLabel(content_frame,
                    text=feature["desc"],
                    font=("Arial", 12),
                    text_color=("#546E7A", "#B0BEC5"),
                    wraplength=250,
                    justify="left").pack(anchor="w")
        
        return card

class ModernContactWindow(ctk.CTkToplevel):
    """Modern Contact Window with Team Information"""
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.title("👥 Contact Our Team")
        self.geometry("900x600")
        self.resizable(False, False)
        
        # Make window modal
        self.transient(parent)
        self.grab_set()
        self.focus_set()
        
        # Bind the close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_modern_contact()
        self.center_window()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_modern_contact(self):
        # Main container
        main_frame = ctk.CTkFrame(self, 
                                 corner_radius=25,
                                 fg_color=("white", "#1E2A38"),
                                 border_width=2,
                                 border_color=("#E3F2FD", "#37474F"))
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(pady=30)

        ctk.CTkLabel(header_frame, 
                    text="👥",
                    font=("Arial", 32)).pack()
        
        ctk.CTkLabel(header_frame,
                    text="Contact Our Team",
                    font=("Arial", 24, "bold"),
                    text_color=("#9C27B0", "#CE93D8")).pack(pady=(10, 5))
        
        ctk.CTkLabel(header_frame,
                    text="Team behind the satellite image analysis system",
                    font=("Arial", 13),
                    text_color=("gray50", "gray70")).pack()

        # Team members container
        team_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        team_frame.pack(pady=20, padx=40, fill="both", expand=True)

        # Team members data
        team_members = [
            {
                "name": "Yashas T G S",
                "email": "tgsyashas@gmail.com",
                "initials": "YT",
                "bg_color": "#E91E63"
            },
            {
                "name": "Ruthu M S",
                "email": "ruthums88@gmail.com",
                "initials": "RM",
                "bg_color": "#2196F3"
            },
            {
                "name": "Sindhu J M",
                "email": "sindhujm17@gmail.com",
                "initials": "SJ",
                "bg_color": "#4CAF50"
            }
        ]

        # Create team member cards
        for i, member in enumerate(team_members):
            member_card = self.create_team_member_card(team_frame, member)
            member_card.grid(row=0, column=i, padx=15, pady=15, sticky="nsew")
            team_frame.grid_columnconfigure(i, weight=1)

        # Back button
        ctk.CTkButton(main_frame,
                     text="← Back to Home",
                     font=("Arial", 12),
                     fg_color="transparent",
                     hover_color=("gray90", "gray30"),
                     command=self.on_closing).pack(side="bottom", pady=20)

    def create_team_member_card(self, parent, member):
        """Create a card for a team member"""
        card = ctk.CTkFrame(parent, 
                           corner_radius=15,
                           fg_color=("#F5F5F5", "#263238"),
                           border_width=1,
                           border_color=("#E0E0E0", "#37474F"))
        card.pack_propagate(False)
        card.configure(width=250, height=350)

        # Photo frame
        photo_frame = ctk.CTkFrame(card, fg_color="transparent", corner_radius=10)
        photo_frame.pack(pady=(20, 10))
        
        # Create a canvas for the photo with initials
        photo_canvas = tk.Canvas(photo_frame, width=150, height=150, bg=member["bg_color"], highlightthickness=0)
        photo_canvas.pack()
        
        # Draw initials on canvas
        initials = member["initials"]
        photo_canvas.create_text(75, 75, text=initials, font=("Arial", 48, "bold"), fill="white")

        # Name
        name_label = ctk.CTkLabel(card,
                                 text=member["name"],
                                 font=("Arial", 16, "bold"),
                                 text_color=("#212121", "#ECEFF1"))
        name_label.pack(pady=(10, 5))

        # Position
        position_label = ctk.CTkLabel(card,
                                     text="Team Member",
                                     font=("Arial", 12),
                                     text_color=("#757575", "#B0BEC5"))
        position_label.pack(pady=(0, 10))

        # Email
        email_label = ctk.CTkLabel(card,
                                  text=member["email"],
                                  font=("Arial", 11),
                                  text_color=("#1976D2", "#4FC3F7"),
                                  wraplength=200)
        email_label.pack(pady=(0, 15))

        # Contact button
        contact_btn = ctk.CTkButton(card,
                                   text="✉️ Email",
                                   font=("Arial", 12, "bold"),
                                   height=30,
                                   width=100,
                                   corner_radius=15,
                                   fg_color="#9C27B0",
                                   hover_color="#7B1FA2",
                                   command=lambda email=member["email"]: self.send_email(email))
        contact_btn.pack(pady=(0, 15))
        
        return card

    def send_email(self, email):
        """Open default email client with pre-filled recipient"""
        try:
            webbrowser.open(f"mailto:{email}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open email client: {str(e)}")

    def on_closing(self):
        self.destroy()

class ModernLoginWindow(ctk.CTkToplevel):
    """Modern Login Window with Glass Morphism Effect"""
    def __init__(self, parent, classifier, app):
        super().__init__(parent)
        self.classifier = classifier
        self.app = app
        self.result = False
        
        self.title("🔐 Login - Satellite Analysis")
        self.geometry("450x600")
        self.resizable(False, False)
        
        # Make window modal
        self.transient(parent)
        self.grab_set()
        self.focus_set()
        
        # Bind the close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_modern_login()
        self.center_window()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_modern_login(self):
        # Main container with glass effect
        main_frame = ctk.CTkFrame(self, 
                                 corner_radius=25,
                                 fg_color=("white", "#1E2A38"),
                                 border_width=2,
                                 border_color=("#E3F2FD", "#37474F"))
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(pady=40)

        ctk.CTkLabel(header_frame, 
                    text="🔐",
                    font=("Arial", 32)).pack()
        
        ctk.CTkLabel(header_frame,
                    text="Welcome Back",
                    font=("Arial", 24, "bold"),
                    text_color=("#1976D2", "#4FC3F7")).pack(pady=(10, 5))
        
        ctk.CTkLabel(header_frame,
                    text="Sign in to access satellite analysis",
                    font=("Arial", 13),
                    text_color=("gray50", "gray70")).pack()

        # Input form
        form_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        form_frame.pack(pady=30, padx=40, fill="both", expand=True)

        # Username field
        ctk.CTkLabel(form_frame, 
                    text="Username",
                    font=("Arial", 13, "bold"),
                    text_color=("gray30", "gray70")).pack(anchor="w", pady=(15, 8))
        
        self.username = ctk.CTkEntry(form_frame,
                                    placeholder_text="Enter your username",
                                    height=50,
                                    corner_radius=12,
                                    border_width=2,
                                    font=("Arial", 14))
        self.username.pack(fill="x", pady=(0, 15))
        self.username.bind('<Return>', lambda e: self.authenticate())

        # Password field
        ctk.CTkLabel(form_frame,
                    text="Password", 
                    font=("Arial", 13, "bold"),
                    text_color=("gray30", "gray70")).pack(anchor="w", pady=(5, 8))
        
        self.password = ctk.CTkEntry(form_frame,
                                    placeholder_text="Enter your password",
                                    show="•",
                                    height=50,
                                    corner_radius=12,
                                    border_width=2,
                                    font=("Arial", 14))
        self.password.pack(fill="x", pady=(0, 20))
        self.password.bind('<Return>', lambda e: self.authenticate())

        # Login button
        login_btn = ctk.CTkButton(form_frame,
                                 text="Sign In",
                                 command=self.authenticate,
                                 height=50,
                                 corner_radius=12,
                                 font=("Arial", 16, "bold"),
                                 fg_color="#1976D2",
                                 hover_color="#1565C0")
        login_btn.pack(fill="x", pady=(10, 15))

        # Register link
        register_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        register_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(register_frame,
                    text="Don't have an account?",
                    font=("Arial", 12),
                    text_color=("gray50", "gray70")).pack(side="left")
        
        ctk.CTkButton(register_frame,
                     text="Create one",
                     font=("Arial", 12, "bold"),
                     fg_color="transparent",
                     hover_color=("gray90", "gray30"),
                     text_color=("#1976D2", "#4FC3F7"),
                     command=self.show_register).pack(side="left", padx=5)

        # Back button
        ctk.CTkButton(main_frame,
                     text="← Back to Home",
                     font=("Arial", 12),
                     fg_color="transparent",
                     hover_color=("gray90", "gray30"),
                     command=self.on_closing).pack(side="bottom", pady=20)

    def authenticate(self):
        username = self.username.get().strip()
        password = self.password.get()

        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return

        try:
            if self.classifier.authenticate_user(username, password):
                self.result = True
                self.destroy()
                self.app.show_main_application()
        except AuthenticationError as e:
            messagebox.showerror("Authentication Failed", str(e))
            self.password.delete(0, tk.END)

    def show_register(self):
        self.destroy()
        self.app.show_register()

    def on_closing(self):
        self.destroy()

class ModernRegisterWindow(ctk.CTkToplevel):
    """Modern Registration Window"""
    def __init__(self, parent, classifier, app):
        super().__init__(parent)
        self.classifier = classifier
        self.app = app
        self.result = False
        
        self.title("📝 Register - Satellite Analysis")
        self.geometry("500x700")
        self.resizable(False, False)
        
        self.transient(parent)
        self.grab_set()
        self.focus_set()
        
        # Bind the close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_modern_register()
        self.center_window()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_modern_register(self):
        # Main container
        main_frame = ctk.CTkFrame(self,
                                 corner_radius=25,
                                 fg_color=("white", "#1E2A38"),
                                 border_width=2,
                                 border_color=("#E3F2FD", "#37474F"))
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(pady=30)

        ctk.CTkLabel(header_frame,
                    text="📝",
                    font=("Arial", 32)).pack()
        
        ctk.CTkLabel(header_frame,
                    text="Create Account",
                    font=("Arial", 24, "bold"),
                    text_color=("#388E3C", "#81C784")).pack(pady=(10, 5))
        
        ctk.CTkLabel(header_frame,
                    text="Join our Satellite analysis platform",
                    font=("Arial", 13),
                    text_color=("gray50", "gray70")).pack()

        # Form
        form_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        form_frame.pack(pady=20, padx=40, fill="both", expand=True)

        # Username
        ctk.CTkLabel(form_frame,
                    text="Username",
                    font=("Arial", 13, "bold")).pack(anchor="w", pady=(10, 5))
        
        self.username = ctk.CTkEntry(form_frame,
                                    placeholder_text="Choose a username",
                                    height=45,
                                    corner_radius=10)
        self.username.pack(fill="x", pady=(0, 10))

        # Email
        ctk.CTkLabel(form_frame,
                    text="Email (Optional)",
                    font=("Arial", 13, "bold")).pack(anchor="w", pady=(5, 5))
        
        self.email = ctk.CTkEntry(form_frame,
                                 placeholder_text="your.email@example.com",
                                 height=45,
                                 corner_radius=10)
        self.email.pack(fill="x", pady=(0, 10))

        # Password
        ctk.CTkLabel(form_frame,
                    text="Password",
                    font=("Arial", 13, "bold")).pack(anchor="w", pady=(5, 5))
        
        self.password = ctk.CTkEntry(form_frame,
                                    placeholder_text="Create a strong password",
                                    show="•",
                                    height=45,
                                    corner_radius=10)
        self.password.pack(fill="x", pady=(0, 10))

        # Confirm Password
        ctk.CTkLabel(form_frame,
                    text="Confirm Password",
                    font=("Arial", 13, "bold")).pack(anchor="w", pady=(5, 5))
        
        self.confirm_password = ctk.CTkEntry(form_frame,
                                           placeholder_text="Re-enter your password",
                                           show="•",
                                           height=45,
                                           corner_radius=10)
        self.confirm_password.pack(fill="x", pady=(0, 20))

        # Register button
        register_btn = ctk.CTkButton(form_frame,
                                    text="Create Account",
                                    command=self.register,
                                    height=50,
                                    corner_radius=12,
                                    font=("Arial", 16, "bold"),
                                    fg_color="#388E3C",
                                    hover_color="#2E7D32")
        register_btn.pack(fill="x", pady=(10, 15))

        # Login link
        login_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        login_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(login_frame,
                    text="Already have an account?",
                    font=("Arial", 12)).pack(side="left")
        
        ctk.CTkButton(login_frame,
                     text="Sign in",
                     font=("Arial", 12, "bold"),
                     fg_color="transparent",
                     hover_color=("gray90", "gray30"),
                     text_color=("#1976D2", "#4FC3F7"),
                     command=self.show_login).pack(side="left", padx=5)

        # Back button
        ctk.CTkButton(main_frame,
                     text="← Back to Home",
                     font=("Arial", 12),
                     fg_color="transparent",
                     hover_color=("gray90", "gray30"),
                     command=self.on_closing).pack(side="bottom", pady=20)

    def register(self):
        username = self.username.get().strip()
        email = self.email.get().strip()
        password = self.password.get()
        confirm_password = self.confirm_password.get()

        if not username or not password:
            messagebox.showerror("Error", "Username and password are required")
            return

        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match")
            return

        try:
            if self.classifier.register_user(username, password, email or None):
                messagebox.showinfo("Success", "Registration successful! Please login.")
                self.destroy()
                self.app.show_login()
        except RegistrationError as e:
            messagebox.showerror("Registration Failed", str(e))

    def show_login(self):
        self.destroy()
        self.app.show_login()

    def on_closing(self):
        self.destroy()

class ModernScrollableImage(ctk.CTkFrame):
    """Modern Scrollable Image Canvas"""
    def __init__(self, parent, title):
        super().__init__(parent)
        self.title = title
        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = ctk.CTkLabel(self, 
                                  text=self.title,
                                  font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))

        # Canvas frame
        canvas_frame = ctk.CTkFrame(self, corner_radius=10)
        canvas_frame.pack(fill="both", expand=True)

        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", highlightthickness=0)
        
        h_scroll = ctk.CTkScrollbar(canvas_frame, orientation="horizontal", command=self.canvas.xview)
        v_scroll = ctk.CTkScrollbar(canvas_frame, orientation="vertical", command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.image_obj = None

    def display_image(self, pil_image):
        self.photo = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.image_obj = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

class QuadrantDisplayWindow(ctk.CTkToplevel):
    """Window to display a quadrant of the satellite image with classification"""
    def __init__(self, parent, app, quadrant_num, original_img, processed_img, location_info):
        super().__init__(parent)
        self.app = app
        self.quadrant_num = quadrant_num
        self.location_info = location_info
        
        self.title(f"🛰️ Quadrant {quadrant_num} Classification")
        self.geometry("1000x700")
        
        # Make window modal
        self.transient(parent)
        self.grab_set()
        self.focus_set()
        
        # Center the window
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        
        # Store the images
        self.original_img = original_img
        self.processed_img = processed_img
        
        # Create the UI
        self.create_widgets()
        
        # Display the images
        self.display_images()

    def create_widgets(self):
        # Header
        header = ctk.CTkFrame(self, height=70, corner_radius=0,
                             fg_color=("#0D47A1", "#1A237E"))
        header.pack(fill="x", padx=0, pady=0)
        
        header_content = ctk.CTkFrame(header, fg_color="transparent")
        header_content.pack(fill="both", padx=30, pady=10)

        ctk.CTkLabel(header_content,
                    text=f"🛰️ Quadrant {self.quadrant_num} Classification",
                    font=("Arial", 20, "bold"),
                    text_color="white").pack(side="left")

        # Close button
        close_button = ctk.CTkButton(
            header_content,
            text="✕ Close All",
            command=self.close_all_windows,
            width=120,
            height=35,
            fg_color="#F44336",
            hover_color="#D32F2F"
        )
        close_button.pack(side="right")
        
        # Location info
        if self.location_info:
            location_label = ctk.CTkLabel(
                self,
                text=f"📍 Location: {self.location_info}",
                font=("Arial", 12),
                text_color=("#1976D2", "#4FC3F7")
            )
            location_label.pack(pady=(10, 5))
        
        # Image display area
        display_frame = ctk.CTkFrame(self, fg_color="transparent")
        display_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Original image panel
        left_panel = ctk.CTkFrame(display_frame, corner_radius=15)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(left_panel, 
                    text="📷 Original Quadrant",
                    font=("Arial", 16, "bold")).pack(pady=(10, 5))
        
        self.orig_canvas = ModernScrollableImage(left_panel, "")
        self.orig_canvas.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Processed image panel
        right_panel = ctk.CTkFrame(display_frame, corner_radius=15)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        ctk.CTkLabel(right_panel,
                    text="🎨 Classified Quadrant",
                    font=("Arial", 16, "bold")).pack(pady=(10, 5))
        
        self.proc_canvas = ModernScrollableImage(right_panel, "")
        self.proc_canvas.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Legend frame
        self.legend_frame = ctk.CTkFrame(self, corner_radius=10)
        self.legend_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Status bar
        self.status_var = ctk.StringVar(value="🟢 Ready")
        status_bar = ctk.CTkLabel(self,
                                 textvariable=self.status_var,
                                 font=("Arial", 12),
                                 corner_radius=0,
                                 fg_color=("gray85", "gray25"))
        status_bar.pack(side="bottom", fill="x")

    def display_images(self):
        # Display original image
        if self.original_img is not None:
            display_img = ImageOps.contain(self.original_img, (400, 400))
            self.orig_canvas.display_image(display_img)
        
        # Display processed image
        if self.processed_img is not None:
            display_img = ImageOps.contain(self.processed_img, (400, 400))
            self.proc_canvas.display_image(display_img)
            
            # Show legend and analysis
            self.show_legend_and_analysis()

    def show_legend_and_analysis(self):
        # Clear previous legend
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        # Check if we have valid data
        if self.processed_img is None:
            self.status_var.set("❌ No processed image available")
            return
            
        if not hasattr(self.app.classifier, 'class_names') or not hasattr(self.app.classifier, 'class_colors'):
            self.status_var.set("❌ Classifier not properly initialized")
            return

        class_names = self.app.classifier.class_names
        class_colors = self.app.classifier.class_colors

        # Count classes in the processed image
        try:
            h, w, _ = self.processed_img.shape
        except (ValueError, AttributeError) as e:
            self.status_var.set(f"❌ Invalid image data: {str(e)}")
            return
            
        grid_size = self.app.classifier.grid_size
        current_counts = {name: 0 for name in class_names}

        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                try:
                    grid_color = self.processed_img[y:y+grid_size, x:x+grid_size].mean(axis=(0, 1))
                    # Find closest color
                    min_dist = float('inf')
                    label = None
                    for idx, color in class_colors.items():
                        dist = np.linalg.norm(grid_color - np.array(color))
                        if dist < min_dist:
                            min_dist = dist
                            label = class_names[idx]
                    if label:
                        current_counts[label] += 1
                except Exception as e:
                    logger.warning(f"Error processing grid at ({x}, {y}): {str(e)}")
                    continue

        # Create modern legend
        legend_title = ctk.CTkLabel(self.legend_frame,
                                   text="🗺️ Classification Legend",
                                   font=("Arial", 14, "bold"))
        legend_title.pack(pady=(10, 5))

        legend_content = ctk.CTkFrame(self.legend_frame, fg_color="transparent")
        legend_content.pack(fill="x", padx=20, pady=10)

        # Create legend items in a grid
        for i, name in enumerate(class_names):
            row = i // 5
            col = i % 5
            
            item_frame = ctk.CTkFrame(legend_content, fg_color="transparent")
            item_frame.grid(row=row, column=col, padx=10, pady=5, sticky="w")
            
            color_hex = f"#{class_colors[i][0]:02x}{class_colors[i][1]:02x}{class_colors[i][2]:02x}"
            color_label = tk.Label(item_frame, bg=color_hex, width=4, height=1,
                                 relief="solid", borderwidth=1)
            color_label.pack(side="left", padx=(0, 5))
            
            text = f"{name}: {current_counts[name]}"
            text_label = ctk.CTkLabel(item_frame, text=text, font=("Arial", 11))
            text_label.pack(side="left")
        
        # Update status
        self.status_var.set("✅ Classification complete")

    def close_all_windows(self):
        """Close all quadrant windows and return to the main dashboard"""
        # Close all quadrant windows
        for window in self.app.quadrant_windows:
            if window.winfo_exists():
                window.destroy()
        
        # Clear the list
        self.app.quadrant_windows = []
        
        # Close the parent image window if it exists
        if hasattr(self.app, 'satellite_image_window') and self.app.satellite_image_window.winfo_exists():
            self.app.satellite_image_window.destroy()
        
        # Show the main dashboard
        self.app.show_main_application()

class ModernMainApplication(ctk.CTkFrame):
    """Modern Main Application Dashboard"""
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        # Header with gradient effect
        header = ctk.CTkFrame(self, height=70, corner_radius=0,
                             fg_color=("#0D47A1", "#1A237E"))
        header.pack(fill="x", padx=0, pady=0)
        
        header_content = ctk.CTkFrame(header, fg_color="transparent")
        header_content.pack(fill="both", padx=30, pady=10)

        ctk.CTkLabel(header_content,
                    text="🛰️ Satellite Image Classification Dashboard",
                    font=("Arial", 20, "bold"),
                    text_color="white").pack(side="left")

        ctk.CTkButton(header_content,
                     text="🚪 Logout",
                     command=self.app.logout,
                     width=100,
                     height=35,
                     fg_color="transparent",
                     border_width=2,
                     border_color="white",
                     hover_color="#1565C0").pack(side="right")

        # Main content area
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.pack(fill="both", expand=True, padx=20, pady=20)

        # Control panel - Modern card design
        control_card = ctk.CTkFrame(main_content, corner_radius=15)
        control_card.pack(fill="x", pady=(0, 20))
        control_card.pack_propagate(False)
        control_card.configure(height=120)

        control_content = ctk.CTkFrame(control_card, fg_color="transparent")
        control_content.pack(fill="both", padx=20, pady=15)

        # Model info
        self.model_var = ctk.StringVar(value="No model loaded")
        model_label = ctk.CTkLabel(control_content,
                                  textvariable=self.model_var,
                                  font=("Arial", 14, "bold"),
                                  text_color=("#1976D2", "#4FC3F7"))
        model_label.pack(side="left", padx=10)

        # Action buttons
        button_frame = ctk.CTkFrame(control_content, fg_color="transparent")
        button_frame.pack(side="right")

        ctk.CTkButton(button_frame,
                     text="📁 Load Model",
                     command=self.app.load_model_dialog,
                     width=120,
                     height=40,
                     corner_radius=10).pack(side="left", padx=5)

        self.load_image_btn = ctk.CTkButton(button_frame,
                                          text="🖼️ Load Image",
                                          command=self.app.load_image_dialog,
                                          width=120,
                                          height=40,
                                          corner_radius=10,
                                          state="disabled")
        self.load_image_btn.pack(side="left", padx=5)

        self.process_btn = ctk.CTkButton(button_frame,
                                       text="⚡ Process",
                                       command=self.app.process_image,
                                       width=120,
                                       height=40,
                                       corner_radius=10,
                                       fg_color="#388E3C",
                                       hover_color="#2E7D32",
                                       state="disabled")
        self.process_btn.pack(side="left", padx=5)

        # Image display area
        display_frame = ctk.CTkFrame(main_content, fg_color="transparent")
        display_frame.pack(fill="both", expand=True)

        # Original image panel
        left_panel = ctk.CTkFrame(display_frame, corner_radius=15)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.orig_canvas = ModernScrollableImage(left_panel, "📷 Original Satellite Image")
        self.orig_canvas.pack(fill="both", expand=True, padx=15, pady=15)

        # Processed image panel
        right_panel = ctk.CTkFrame(display_frame, corner_radius=15)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        self.proc_canvas = ModernScrollableImage(right_panel, "🎨 Classified Image")
        self.proc_canvas.pack(fill="both", expand=True, padx=15, pady=15)

        # Analysis buttons frame
        analysis_frame = ctk.CTkFrame(main_content, fg_color="transparent")
        analysis_frame.pack(fill="x", pady=(20, 0))
        
        # Create a frame for the buttons to place them side by side
        buttons_container = ctk.CTkFrame(analysis_frame, fg_color="transparent")
        buttons_container.pack(pady=5)
        
        # Visualize button
        self.visualize_btn = ctk.CTkButton(buttons_container,
                                         text="📊 Visualize Analysis",
                                         command=self.app.show_visualization,
                                         width=200,
                                         height=40,
                                         corner_radius=10,
                                         fg_color="#7B1FA2",
                                         hover_color="#6A1B9A",
                                         state="disabled")
        self.visualize_btn.pack(side="left", padx=5)
        
        # Classification button
        self.classification_btn = ctk.CTkButton(buttons_container,
                                              text="🥧 Classification Chart",
                                              command=self.app.show_classification_pie_chart,
                                              width=200,
                                              height=40,
                                              corner_radius=10,
                                              fg_color="#00897B",
                                              hover_color="#00796B",
                                              state="disabled")
        self.classification_btn.pack(side="left", padx=5)
        
        # Data of Image button
        self.data_btn = ctk.CTkButton(buttons_container,
                                     text="📋 Data of Image",
                                     command=self.app.show_image_data,
                                     width=200,
                                     height=40,
                                     corner_radius=10,
                                     fg_color="#5D4037",
                                     hover_color="#4E342E",
                                     state="disabled")
        self.data_btn.pack(side="left", padx=5)
        
        # Analyse button (NEW)
        self.analyse_btn = ctk.CTkButton(buttons_container,
                                        text="🔍 Analyse",
                                        command=self.app.show_analyse_window,
                                        width=200,
                                        height=40,
                                        corner_radius=10,
                                        fg_color="#FF9800",
                                        hover_color="#F57C00")
        self.analyse_btn.pack(side="left", padx=5)

        # Legend and status area
        bottom_frame = ctk.CTkFrame(main_content, fg_color="transparent")
        bottom_frame.pack(fill="x", pady=(20, 0))

        self.legend_frame = ctk.CTkFrame(bottom_frame, corner_radius=10)
        self.legend_frame.pack(fill="x", pady=5)

        # Status bar
        self.status_var = ctk.StringVar(value="🟢 Ready - Load a model and image to start analysis")
        status_bar = ctk.CTkLabel(self,
                                 textvariable=self.status_var,
                                 font=("Arial", 12),
                                 corner_radius=0,
                                 fg_color=("gray85", "gray25"))
        status_bar.pack(side="bottom", fill="x")

class QuadrantAnalysisWindow(ctk.CTkToplevel):
    """Window to analyze a single quadrant image"""
    def __init__(self, parent, app, quadrant_num):
        super().__init__(parent)
        self.app = app
        self.quadrant_num = quadrant_num
        self.image_path = None
        self.original_img = None
        self.classified_img = None
        self.classifier = app.classifier
        
        self.title(f"🔍 Analyze Quadrant {quadrant_num}")
        self.geometry("1000x700")
        
        # Make window modal
        self.transient(parent)
        self.grab_set()
        self.focus_set()
        
        # Center the window
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        
        # Create the UI
        self.create_widgets()
        
        # Add to quadrant windows list
        self.app.quadrant_windows.append(self)

    def create_widgets(self):
        # Header
        header = ctk.CTkFrame(self, height=70, corner_radius=0,
                             fg_color=("#0D47A1", "#1A237E"))
        header.pack(fill="x", padx=0, pady=0)
        
        header_content = ctk.CTkFrame(header, fg_color="transparent")
        header_content.pack(fill="both", padx=30, pady=10)

        ctk.CTkLabel(header_content,
                    text=f"🔍 Analyze Quadrant {self.quadrant_num}",
                    font=("Arial", 20, "bold"),
                    text_color="white").pack(side="left")

        # Close button
        close_button = ctk.CTkButton(
            header_content,
            text="✕ Cancel",
            command=self.cancel_analysis,
            width=100,
            height=35,
            fg_color="#F44336",
            hover_color="#D32F2F"
        )
        close_button.pack(side="right")
        
        # Main content area
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Instructions
        instructions = ctk.CTkLabel(
            main_content,
            text=f"Select and process the image for Quadrant {self.quadrant_num}",
            font=("Arial", 14),
            text_color=("#1976D2", "#4FC3F7")
        )
        instructions.pack(pady=(10, 20))
        
        # Image selection area
        selection_frame = ctk.CTkFrame(main_content, corner_radius=10)
        selection_frame.pack(fill="x", pady=(0, 20))
        
        # Select button
        select_btn = ctk.CTkButton(
            selection_frame,
            text="📁 Select Image",
            command=self.select_image,
            width=150,
            height=40,
            corner_radius=10
        )
        select_btn.pack(pady=10)
        
        # Image preview
        self.preview_canvas = tk.Canvas(selection_frame, bg="#2b2b2b", highlightthickness=0, height=200)
        self.preview_canvas.pack(fill="x", padx=15, pady=10)
        
        # Add placeholder text
        self.preview_canvas.create_text(
            self.preview_canvas.winfo_width() // 2,
            self.preview_canvas.winfo_height() // 2,
            text="No image selected",
            fill="white",
            font=("Arial", 12)
        )
        
        # Process button
        self.process_btn = ctk.CTkButton(
            selection_frame,
            text="⚡ Process Image",
            command=self.process_image,
            width=150,
            height=40,
            corner_radius=10,
            fg_color="#388E3C",
            hover_color="#2E7D32",
            state="disabled"
        )
        self.process_btn.pack(pady=10)
        
        # Results area (initially hidden)
        self.results_frame = ctk.CTkFrame(main_content, fg_color="transparent")
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(main_content, fg_color="transparent")
        nav_frame.pack(fill="x", pady=20)
        
        # Next/Close button
        if self.quadrant_num < 4:
            self.next_btn = ctk.CTkButton(
                nav_frame,
                text="➡️ Next Quadrant",
                command=self.next_quadrant,
                width=150,
                height=40,
                corner_radius=10,
                fg_color="#2196F3",
                hover_color="#1976D2",
                state="disabled"
            )
            self.next_btn.pack(side="right", padx=10)
        else:
            self.close_all_btn = ctk.CTkButton(
                nav_frame,
                text="✕ Close All",
                command=self.close_all_windows,
                width=150,
                height=40,
                corner_radius=10,
                fg_color="#F44336",
                hover_color="#D32F2F",
                state="disabled"
            )
            self.close_all_btn.pack(side="right", padx=10)
        
        # Status bar
        self.status_var = ctk.StringVar(value="🟢 Ready - Select an image to analyze")
        status_bar = ctk.CTkLabel(self,
                                 textvariable=self.status_var,
                                 font=("Arial", 12),
                                 corner_radius=0,
                                 fg_color=("gray85", "gray25"))
        status_bar.pack(side="bottom", fill="x")

    def select_image(self):
        path = filedialog.askopenfilename(
            title=f"Select Image for Quadrant {self.quadrant_num}",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if path:
            self.image_path = Path(path)
            
            # Load and display preview
            try:
                pil_img = Image.open(path)
                # Resize for preview
                preview_img = ImageOps.contain(pil_img, (600, 200))
                
                # Display the image
                self.preview_canvas.delete("all")
                self.preview_canvas.preview_photo = ImageTk.PhotoImage(preview_img)
                self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_canvas.preview_photo)
                
                self.process_btn.configure(state="normal")
                self.status_var.set(f"✅ Image selected: {self.image_path.name}")
            except Exception as e:
                messagebox.showerror("Image Load Error", f"Failed to load image: {str(e)}")
                self.image_path = None

    def process_image(self):
        if not self.image_path:
            return
        
        # Process image in a thread
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        try:
            self.status_var.set("🔄 Processing image... This may take a moment.")
            
            # Process the image
            orig, colored = self.classifier.process_image(str(self.image_path))
            
            # Check if processing returned valid results
            if orig is None or colored is None:
                raise ImageProcessingError("Image processing returned invalid results")
            
            # Convert to displayable format
            self.original_img = Image.fromarray((orig * 255).astype(np.uint8))
            self.classified_img = Image.fromarray((colored * 255).astype(np.uint8))
            
            # Display results
            self.display_results()
            
            self.status_var.set("✅ Image processed successfully!")
            
            # Enable next/close button
            if self.quadrant_num < 4:
                self.next_btn.configure(state="normal")
            else:
                self.close_all_btn.configure(state="normal")
            
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            self.status_var.set("❌ Image processing failed")

    def display_results(self):
        # Show the results frame
        self.results_frame.pack(fill="both", expand=True, pady=20)
        
        # Title
        results_title = ctk.CTkLabel(
            self.results_frame,
            text=f"📊 Quadrant {self.quadrant_num} Analysis Results",
            font=("Arial", 18, "bold")
        )
        results_title.pack(pady=(10, 20))
        
        # Create a frame for the images
        images_frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        images_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Original image panel
        left_panel = ctk.CTkFrame(images_frame, corner_radius=15)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(left_panel, 
                    text="📷 Original Image",
                    font=("Arial", 16, "bold")).pack(pady=(10, 5))
        
        # Resize for display
        display_orig = ImageOps.contain(self.original_img, (400, 300))
        
        orig_canvas = tk.Canvas(left_panel, bg="#2b2b2b", highlightthickness=0, width=400, height=300)
        orig_canvas.pack(padx=15, pady=15)
        
        orig_canvas.orig_photo = ImageTk.PhotoImage(display_orig)
        orig_canvas.create_image(0, 0, anchor="nw", image=orig_canvas.orig_photo)
        
        # Classified image panel
        right_panel = ctk.CTkFrame(images_frame, corner_radius=15)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        ctk.CTkLabel(right_panel,
                    text="🎨 Classified Image",
                    font=("Arial", 16, "bold")).pack(pady=(10, 5))
        
        # Resize for display
        display_class = ImageOps.contain(self.classified_img, (400, 300))
        
        class_canvas = tk.Canvas(right_panel, bg="#2b2b2b", highlightthickness=0, width=400, height=300)
        class_canvas.pack(padx=15, pady=15)
        
        class_canvas.class_photo = ImageTk.PhotoImage(display_class)
        class_canvas.create_image(0, 0, anchor="nw", image=class_canvas.class_photo)
        
        # Add classification legend
        legend_frame = ctk.CTkFrame(self.results_frame, corner_radius=10)
        legend_frame.pack(fill="x", pady=20)
        
        legend_title = ctk.CTkLabel(legend_frame,
                                   text="🗺️ Classification Legend",
                                   font=("Arial", 14, "bold"))
        legend_title.pack(pady=(10, 5))
        
        legend_content = ctk.CTkFrame(legend_frame, fg_color="transparent")
        legend_content.pack(fill="x", padx=20, pady=10)
        
        # Get class names and colors
        if hasattr(self.classifier, 'class_names') and hasattr(self.classifier, 'class_colors'):
            class_names = self.classifier.class_names
            class_colors = self.classifier.class_colors
            
            # Create legend items in a grid
            for i, name in enumerate(class_names):
                row = i // 5
                col = i % 5
                
                item_frame = ctk.CTkFrame(legend_content, fg_color="transparent")
                item_frame.grid(row=row, column=col, padx=10, pady=5, sticky="w")
                
                color_hex = f"#{class_colors[i][0]:02x}{class_colors[i][1]:02x}{class_colors[i][2]:02x}"
                color_label = tk.Label(item_frame, bg=color_hex, width=4, height=1,
                                     relief="solid", borderwidth=1)
                color_label.pack(side="left", padx=(0, 5))
                
                text_label = ctk.CTkLabel(item_frame, text=name, font=("Arial", 11))
                text_label.pack(side="left")

    def next_quadrant(self):
        # Close current window
        self.destroy()
        
        # Open next quadrant window
        next_window = QuadrantAnalysisWindow(self.app, self.app, self.quadrant_num + 1)
        self.app.wait_window(next_window)

    def close_all_windows(self):
        # Close all quadrant windows
        for window in self.app.quadrant_windows:
            if window.winfo_exists():
                window.destroy()
        
        # Clear the list
        self.app.quadrant_windows = []
        
        # Show the main dashboard
        self.app.show_main_application()

    def cancel_analysis(self):
        # Close current window
        self.destroy()
        
        # If this is the first quadrant, return to main dashboard
        if self.quadrant_num == 1:
            self.app.show_main_application()

class ModernSatelliteApp(ctk.CTk):
    """Modern Main Application"""
    def __init__(self):
        super().__init__()
        self.classifier = SatelliteImageClassifier(authentication_enabled=True)
        self.current_image_path = None
        self.original_image = None
        self.colored_image = None
        self.current_counts = None  # Store classification counts for visualization
        self.image_metadata = {}  # Store image metadata
        self.main_app = None  # Store reference to main application
        self.quadrant_windows = []  # Store quadrant windows
        self.satellite_image_window = None  # Store satellite image window
        
        self.title("🛰️ Satellite Image Classification System")
        self.geometry("1400x900")
        
        # Center the window
        self.center_window()
        
        # Configure modern theme
        self.setup_modern_theme()
        
        # Show home page
        self.show_home_page()

    def center_window(self):
        self.update_idletasks()
        width, height = 1400, 900
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_modern_theme(self):
        # Set dark theme with blue accent
        ctk.set_appearance_mode("Dark")
        
    def show_home_page(self):
        self.clear_window()
        self.home_page = ModernHomePage(self, self)
        self.home_page.pack(fill="both", expand=True)
        self.update()

    def show_login(self):
        login = ModernLoginWindow(self, self.classifier, self)
        self.wait_window(login)

    def show_register(self):
        register = ModernRegisterWindow(self, self.classifier, self)
        self.wait_window(register)
        
    def show_contact(self):
        contact = ModernContactWindow(self, self)
        self.wait_window(contact)

    def show_main_application(self):
        self.clear_window()
        self.main_app = ModernMainApplication(self, self)
        self.main_app.pack(fill="both", expand=True)
        self.update()  # Force update to ensure proper rendering
        
        # Restore the image data if available
        if self.original_image is not None and self.colored_image is not None:
            # Display original image
            orig_img = Image.fromarray((self.original_image * 255).astype(np.uint8))
            display_img = ImageOps.contain(orig_img, (800, 600))
            self.main_app.orig_canvas.display_image(display_img)
            
            # Display processed image
            proc_img = Image.fromarray(self.colored_image)
            display_img = ImageOps.contain(proc_img, (800, 600))
            self.main_app.proc_canvas.display_image(display_img)
            
            # Restore legend and analysis
            self.show_legend_and_analysis()
            
            # Update status
            self.update_status("✅ Dashboard restored with previous analysis results")

    def logout(self):
        self.classifier.logout()
        self.show_home_page()

    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

    def update_status(self, msg):
        if hasattr(self, 'main_app') and self.main_app:
            self.main_app.status_var.set(msg)
        self.update_idletasks()

    def load_model_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Keras/H5 Models", "*.h5 *.keras"), ("All files", "*.*")]
        )
        if path:
            threading.Thread(target=self._load_model_thread, args=(path,), daemon=True).start()

    def _load_model_thread(self, path):
        try:
            self.update_status("🔄 Loading model...")
            self.classifier.load_model(path)
            model_name = Path(path).name
            self.main_app.model_var.set(f"✅ Model: {model_name}")
            self.main_app.load_image_btn.configure(state="normal")
            self.update_status(f"✅ Model '{model_name}' loaded successfully")
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            self.update_status("❌ Model loading failed")

    def load_image_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Satellite Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.current_image_path = Path(path)
            self.main_app.process_btn.configure(state="normal")
            
            # Load and display image
            pil_img = Image.open(path)
            # Resize for display while maintaining aspect ratio
            display_img = ImageOps.contain(pil_img, (800, 600))
            self.main_app.orig_canvas.display_image(display_img)
            
            # Extract metadata from the image
            self.extract_image_metadata()
            
            self.update_status(f"✅ Image loaded: {self.current_image_path.name}")

    def extract_image_metadata(self):
        """Extract metadata from the loaded image including geolocation if available"""
        if not self.current_image_path:
            return
            
        try:
            # Reset metadata
            self.image_metadata = {}
            
            # Open image and get basic info
            img = Image.open(self.current_image_path)
            self.image_metadata['width'] = img.width
            self.image_metadata['height'] = img.height
            self.image_metadata['format'] = img.format or "Unknown"
            self.image_metadata['mode'] = img.mode or "Unknown"
            
            # Get file info
            file_stat = os.stat(self.current_image_path)
            self.image_metadata['file_size'] = file_stat.st_size
            self.image_metadata['created_time'] = datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            self.image_metadata['modified_time'] = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract GPS coordinates if available
            lat, lon = get_exif_location(self.current_image_path)
            if lat is not None and lon is not None:
                self.image_metadata['latitude'] = lat
                self.image_metadata['longitude'] = lon
                self.update_status(f"✅ Image loaded with GPS coordinates: {lat:.6f}, {lon:.6f}")
            else:
                self.image_metadata['latitude'] = None
                self.image_metadata['longitude'] = None
                self.update_status(f"✅ Image loaded: {self.current_image_path.name} (No GPS data found)")
                
        except Exception as e:
            logger.warning(f"Error extracting image metadata: {str(e)}")
            self.image_metadata = {}

    def process_image(self):
        if not self.current_image_path:
            return
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        try:
            self.update_status("🔄 Processing image... This may take a moment.")
            orig, colored = self.classifier.process_image(str(self.current_image_path))
            
            # Check if processing returned valid results
            if orig is None or colored is None:
                raise ImageProcessingError("Image processing returned invalid results")
                
            self.original_image = orig
            self.colored_image = (colored * 255).astype(np.uint8)

            # Display processed image
            img = Image.fromarray(self.colored_image)
            display_img = ImageOps.contain(img, (800, 600))
            self.main_app.proc_canvas.display_image(display_img)

            # Show legend and analysis
            self.show_legend_and_analysis()
            self.update_status("✅ Image processed successfully! Check the classified results.")
            
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            self.update_status("❌ Image processing failed")

    def show_legend_and_analysis(self):
        # Clear previous legend
        for widget in self.main_app.legend_frame.winfo_children():
            widget.destroy()

        # Check if we have valid data
        if self.colored_image is None:
            self.update_status("❌ No processed image available")
            return
            
        if not hasattr(self.classifier, 'class_names') or not hasattr(self.classifier, 'class_colors'):
            self.update_status("❌ Classifier not properly initialized")
            return

        class_names = self.classifier.class_names
        class_colors = self.classifier.class_colors

        # Count classes in the processed image
        try:
            h, w, _ = self.colored_image.shape
        except (ValueError, AttributeError) as e:
            self.update_status(f"❌ Invalid image data: {str(e)}")
            return
            
        grid_size = self.classifier.grid_size
        self.current_counts = {name: 0 for name in class_names}  # Store counts for visualization

        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                try:
                    grid_color = self.colored_image[y:y+grid_size, x:x+grid_size].mean(axis=(0, 1))
                    # Find closest color
                    min_dist = float('inf')
                    label = None
                    for idx, color in class_colors.items():
                        dist = np.linalg.norm(grid_color - np.array(color))
                        if dist < min_dist:
                            min_dist = dist
                            label = class_names[idx]
                    if label:
                        self.current_counts[label] += 1
                except Exception as e:
                    logger.warning(f"Error processing grid at ({x}, {y}): {str(e)}")
                    continue

        # Create modern legend
        legend_title = ctk.CTkLabel(self.main_app.legend_frame,
                                   text="🗺️ Classification Legend",
                                   font=("Arial", 14, "bold"))
        legend_title.pack(pady=(10, 5))

        legend_content = ctk.CTkFrame(self.main_app.legend_frame, fg_color="transparent")
        legend_content.pack(fill="x", padx=20, pady=10)

        # Create legend items in a grid
        for i, name in enumerate(class_names):
            row = i // 5
            col = i % 5
            
            item_frame = ctk.CTkFrame(legend_content, fg_color="transparent")
            item_frame.grid(row=row, column=col, padx=10, pady=5, sticky="w")
            
            color_hex = f"#{class_colors[i][0]:02x}{class_colors[i][1]:02x}{class_colors[i][2]:02x}"
            color_label = tk.Label(item_frame, bg=color_hex, width=4, height=1,
                                 relief="solid", borderwidth=1)
            color_label.pack(side="left", padx=(0, 5))
            
            text = f"{name}: {self.current_counts[name]}"
            text_label = ctk.CTkLabel(item_frame, text=text, font=("Arial", 11))
            text_label.pack(side="left")

        # Enable all analysis buttons
        self.main_app.visualize_btn.configure(state="normal")
        self.main_app.classification_btn.configure(state="normal")
        self.main_app.data_btn.configure(state="normal")

    def show_visualization(self):
        """Show the land cover classification distribution chart"""
        if not self.current_counts:
            messagebox.showwarning("No Data", "Please process an image first to generate visualization data.")
            return
        
        try:
            self.plot_analysis_chart(self.current_counts)
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to create visualization: {str(e)}")

    def plot_analysis_chart(self, counts):
        """Create and display the land cover classification distribution chart"""
        plt.figure(figsize=(14, 8))
        
        classes = list(counts.keys())
        values = list(counts.values())
        
        # Ensure we have valid colors
        try:
            colors = [np.array(self.classifier.class_colors[i])/255.0 for i in range(len(classes))]
        except (KeyError, IndexError):
            # Fallback colors if class_colors is not properly set
            colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        
        # Create bar chart
        bars = plt.bar(classes, values, color=colors, edgecolor='black', alpha=0.8, linewidth=1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Customize the chart
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylabel("Number of Grid Segments", fontsize=12, fontweight='bold')
        plt.xlabel("Land Cover Classes", fontsize=12, fontweight='bold')
        plt.title("Land Cover Classification Distribution", fontsize=16, fontweight='bold', pad=20)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Add some statistics
        total_segments = sum(values)
        plt.figtext(0.02, 0.02, f"Total Grid Segments: {total_segments}", 
                   fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Show the plot
        plt.show()

    def show_classification_pie_chart(self):
        """Show a pie chart of the classification results"""
        if not self.current_counts:
            messagebox.showwarning("No Data", "Please process an image first to generate classification data.")
            return
        
        try:
            self.plot_classification_pie_chart(self.current_counts)
        except Exception as e:
            messagebox.showerror("Classification Chart Error", f"Failed to create pie chart: {str(e)}")

    def plot_classification_pie_chart(self, counts):
        """Create and display the classification pie chart"""
        plt.figure(figsize=(12, 10))
        
        classes = list(counts.keys())
        values = list(counts.values())
        
        # Ensure we have valid colors
        try:
            colors = [np.array(self.classifier.class_colors[i])/255.0 for i in range(len(classes))]
        except (KeyError, IndexError):
            # Fallback colors if class_colors is not properly set
            colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        
        # Calculate percentages
        total = sum(values)
        if total == 0:
            messagebox.showwarning("No Data", "No classification data available.")
            return
            
        percentages = [(value / total) * 100 for value in values]
        
        # Create pie chart
        wedges, texts, autotexts = plt.pie(
            percentages, 
            labels=classes, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1},
            textprops={'fontsize': 10}
        )
        
        # Customize text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add title
        plt.title("Land Cover Classification Distribution", fontsize=16, fontweight='bold', pad=20)
        
        # Add legend with more information
        legend_labels = [f"{cls}: {count} grids ({percent:.1f}%)" 
                         for cls, count, percent in zip(classes, values, percentages)]
        plt.legend(
            wedges, 
            legend_labels,
            title="Classification Details",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=10
        )
        
        # Add statistics
        plt.figtext(
            0.5, 0.01, 
            f"Total Grid Segments: {total} | Grid Size: {self.classifier.grid_size}x{self.classifier.grid_size} pixels", 
            ha="center", 
            fontsize=12, 
            style='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8)
        )
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)  # Make room for legend
        
        # Show the plot
        plt.show()

    def show_image_data(self):
        """Show detailed data about the image and classification"""
        if not self.current_image_path or not self.current_counts:
            messagebox.showwarning("No Data", "Please process an image first to generate image data.")
            return
        
        try:
            # Create a new window for image data
            data_window = ctk.CTkToplevel(self)
            data_window.title("📋 Image Data Details")
            
            # Set window size to 70% of screen
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            data_window.geometry(f"{int(screen_width * 0.7)}x{int(screen_height * 0.7)}")
            
            # Make window modal
            data_window.transient(self)
            data_window.grab_set()
            data_window.focus_set()
            
            # Center the window
            data_window.update_idletasks()
            width = data_window.winfo_width()
            height = data_window.winfo_height()
            x = (data_window.winfo_screenwidth() // 2) - (width // 2)
            y = (data_window.winfo_screenheight() // 2) - (height // 2)
            data_window.geometry(f'{width}x{height}+{x}+{y}')
            
            # Create notebook for tabs
            notebook = ttk.Notebook(data_window)
            notebook.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Tab 1: Image Information
            info_frame = ctk.CTkFrame(notebook, fg_color="transparent")
            notebook.add(info_frame, text="Image Information")
            
            # Title
            title_label = ctk.CTkLabel(
                info_frame, 
                text="📋 Image Classification Data",
                font=("Arial", 20, "bold")
            )
            title_label.pack(pady=(10, 20))
            
            # Image information section
            img_info_frame = ctk.CTkFrame(info_frame, corner_radius=10)
            img_info_frame.pack(fill="x", pady=(0, 20), padx=10)
            
            img_info_title = ctk.CTkLabel(
                img_info_frame,
                text="🖼️ Image Information",
                font=("Arial", 16, "bold"),
                anchor="w"
            )
            img_info_title.pack(fill="x", padx=15, pady=(10, 5))
            
            # Get image dimensions
            try:
                img_width = self.image_metadata.get('width', 'Unknown')
                img_height = self.image_metadata.get('height', 'Unknown')
                img_format = self.image_metadata.get('format', 'Unknown')
                img_mode = self.image_metadata.get('mode', 'Unknown')
                file_size = self.image_metadata.get('file_size', 'Unknown')
                created_time = self.image_metadata.get('created_time', 'Unknown')
                modified_time = self.image_metadata.get('modified_time', 'Unknown')
                lat = self.image_metadata.get('latitude', None)
                lon = self.image_metadata.get('longitude', None)
            except Exception as e:
                img_width, img_height = "Unknown", "Unknown"
                img_format = "Unknown"
                img_mode = "Unknown"
                file_size = "Unknown"
                created_time = "Unknown"
                modified_time = "Unknown"
                lat, lon = None, None
                logger.warning(f"Error reading image info: {str(e)}")
            
            # Get grid size
            try:
                grid_size = self.classifier.grid_size
                total_grids = sum(self.current_counts.values()) if self.current_counts else 0
            except Exception:
                grid_size = "Unknown"
                total_grids = "Unknown"
            
            # Create image info labels
            info_items = [
                ("File Name", self.current_image_path.name),
                ("File Path", str(self.current_image_path)),
                ("Image Dimensions", f"{img_width} × {img_height} pixels"),
                ("Image Format", img_format),
                ("Image Mode", img_mode),
                ("File Size", f"{file_size} bytes" if file_size != "Unknown" else "Unknown"),
                ("Created", created_time),
                ("Last Modified", modified_time),
                ("Grid Size", f"{grid_size} × {grid_size} pixels"),
                ("Total Grids", total_grids)
            ]
            
            # Add GPS coordinates if available
            if lat is not None and lon is not None:
                info_items.append(("GPS Latitude", f"{lat:.6f}°"))
                info_items.append(("GPS Longitude", f"{lon:.6f}°"))
            
            for label, value in info_items:
                info_row = ctk.CTkFrame(img_info_frame, fg_color="transparent")
                info_row.pack(fill="x", padx=15, pady=2)
                
                ctk.CTkLabel(
                    info_row,
                    text=f"{label}:",
                    font=("Arial", 12, "bold"),
                    width=150,
                    anchor="w"
                ).pack(side="left")
                
                ctk.CTkLabel(
                    info_row,
                    text=str(value),
                    font=("Arial", 12),
                    anchor="w"
                ).pack(side="left", fill="x", expand=True)
            
            # Current Satellite Image section
            sat_img_frame = ctk.CTkFrame(info_frame, corner_radius=10)
            sat_img_frame.pack(fill="both", expand=True, pady=(0, 20), padx=10)
            
            sat_img_title = ctk.CTkLabel(
                sat_img_frame,
                text="🌍 Current Satellite Image at Location",
                font=("Arial", 16, "bold"),
                anchor="w"
            )
            sat_img_title.pack(fill="x", padx=15, pady=(10, 5))
            
            # Provider info
            provider_info_frame = ctk.CTkFrame(sat_img_frame, fg_color=("gray90", "gray20"), corner_radius=10)
            provider_info_frame.pack(fill="x", padx=15, pady=5)
            
            provider_info_text = (
                "Fetch The satellite Image--"
                "By The co-ordinates"
            )
            
            ctk.CTkLabel(
                provider_info_frame,
                text=provider_info_text,
                font=("Arial", 10),
                wraplength=800,
                justify="left"
            ).pack(padx=10, pady=5)
            
            # Coordinate input frame
            coord_input_frame = ctk.CTkFrame(sat_img_frame, fg_color="transparent")
            coord_input_frame.pack(fill="x", padx=15, pady=5)
            
            # Latitude input
            lat_frame = ctk.CTkFrame(coord_input_frame, fg_color="transparent")
            lat_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(lat_frame, text="Latitude:", font=("Arial", 12, "bold"), width=150, anchor="w").pack(side="left", padx=5)
            sat_lat_entry = ctk.CTkEntry(lat_frame, placeholder_text="e.g., 37.7749", width=200)
            sat_lat_entry.pack(side="left", padx=5)
            
            # Pre-fill with extracted GPS data if available
            if lat is not None:
                sat_lat_entry.insert(0, str(lat))
            
            # Longitude input
            lon_frame = ctk.CTkFrame(coord_input_frame, fg_color="transparent")
            lon_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(lon_frame, text="Longitude:", font=("Arial", 12, "bold"), width=150, anchor="w").pack(side="left", padx=5)
            sat_lon_entry = ctk.CTkEntry(lon_frame, placeholder_text="e.g., -122.4194", width=200)
            sat_lon_entry.pack(side="left", padx=5)
            
            # Pre-fill with extracted GPS data if available
            if lon is not None:
                sat_lon_entry.insert(0, str(lon))
            
            # Zoom level input
            zoom_frame = ctk.CTkFrame(coord_input_frame, fg_color="transparent")
            zoom_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(zoom_frame, text="Zoom Level:", font=("Arial", 12, "bold"), width=150, anchor="w").pack(side="left", padx=5)
            zoom_entry = ctk.CTkEntry(zoom_frame, placeholder_text="e.g., 15 (higher is closer)", width=200)
            zoom_entry.insert(0, "15")  # Default zoom level
            zoom_entry.pack(side="left", padx=5)
            
            # Buttons container
            buttons_container = ctk.CTkFrame(coord_input_frame, fg_color="transparent")
            buttons_container.pack(fill="x", pady=10)
            
            # Fetch button
            fetch_button = ctk.CTkButton(
                buttons_container,
                text="🔍 Fetch Satellite Image",
                command=lambda: self.fetch_satellite_image(
                    sat_lat_entry, sat_lon_entry, zoom_entry, data_window
                ),
                width=200,
                height=30,
                corner_radius=10
            )
            fetch_button.pack(side="left", padx=5)
            
            # Canvas for satellite image
            sat_canvas = tk.Canvas(sat_img_frame, bg="#2b2b2b", highlightthickness=0, height=400)
            sat_canvas.pack(fill="both", expand=True, padx=15, pady=10)
            
            # Initial canvas text
            sat_canvas.create_text(
                sat_canvas.winfo_width() // 2,
                sat_canvas.winfo_height() // 2,
                text="Enter coordinates and click 'Fetch Satellite Image'",
                fill="white",
                font=("Arial", 12)
            )
            
            # If GPS coordinates are available, automatically fetch the satellite image
            if lat is not None and lon is not None:
                self.fetch_satellite_image(sat_lat_entry, sat_lon_entry, zoom_entry, data_window)
            
            # Tab 2: Google Earth Integration
            earth_frame = ctk.CTkFrame(notebook, fg_color="transparent")
            notebook.add(earth_frame, text="Satellite Earth")
            
            # Title for Earth tab
            earth_title = ctk.CTkLabel(
                earth_frame,
                text="🌍 Satellite Earth Integration",
                font=("Arial", 20, "bold")
            )
            earth_title.pack(pady=(10, 20))
            
            # Location input section
            location_input_frame = ctk.CTkFrame(earth_frame, corner_radius=10)
            location_input_frame.pack(fill="x", pady=(0, 20), padx=10)
            
            # Latitude input
            lat_frame = ctk.CTkFrame(location_input_frame, fg_color="transparent")
            lat_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(lat_frame, text="Latitude:", font=("Arial", 14, "bold"), width=150, anchor="w").pack(side="left", padx=10)
            earth_lat_entry = ctk.CTkEntry(lat_frame, placeholder_text="e.g., 37.7749", width=200)
            earth_lat_entry.pack(side="left", padx=10)
            
            # Pre-fill with extracted GPS data if available
            if lat is not None:
                earth_lat_entry.insert(0, str(lat))
            
            # Longitude input
            lon_frame = ctk.CTkFrame(location_input_frame, fg_color="transparent")
            lon_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(lon_frame, text="Longitude:", font=("Arial", 14, "bold"), width=150, anchor="w").pack(side="left", padx=10)
            earth_lon_entry = ctk.CTkEntry(lon_frame, placeholder_text="e.g., -122.4194", width=200)
            earth_lon_entry.pack(side="left", padx=10)
            
            # Pre-fill with extracted GPS data if available
            if lon is not None:
                earth_lon_entry.insert(0, str(lon))
            
            # Altitude input
            alt_frame = ctk.CTkFrame(location_input_frame, fg_color="transparent")
            alt_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(alt_frame, text="Altitude (m):", font=("Arial", 14, "bold"), width=150, anchor="w").pack(side="left", padx=10)
            alt_entry = ctk.CTkEntry(alt_frame, placeholder_text="e.g., 1000", width=200)
            alt_entry.pack(side="left", padx=10)
            
            # Buttons frame
            buttons_frame = ctk.CTkFrame(earth_frame, fg_color="transparent")
            buttons_frame.pack(fill="x", pady=10)
            
            # Open Google Earth button
            def open_google_earth():
                lat = earth_lat_entry.get()
                lon = earth_lon_entry.get()
                alt = alt_entry.get() or "1000"  # Default altitude if not provided
                
                if not lat or not lon:
                    messagebox.showwarning("Missing Information", "Please enter both latitude and longitude.")
                    return
                
                try:
                    lat_val = float(lat)
                    lon_val = float(lon)
                    alt_val = float(alt)
                    
                    # Validate coordinates
                    if not (-90 <= lat_val <= 90) or not (-180 <= lon_val <= 180):
                        messagebox.showerror("Invalid Coordinates", "Latitude must be between -90 and 90, and longitude between -180 and 180.")
                        return
                    
                    # Create Google Earth URL
                    url = f"https://earth.google.com/web/@{lat_val},{lon_val},{alt_val}a,0d,0y,0h,0t"
                    
                    # Open in browser
                    webbrowser.open(url)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid numeric coordinates.")
            
            earth_button = ctk.CTkButton(
                buttons_frame,
                text="🌍 Open in satellite Earth",
                command=open_google_earth,
                width=200,
                height=40,
                corner_radius=10,
                fg_color="#4285F4",  # Google blue
                hover_color="#3367D6"
            )
            earth_button.pack(side="left", padx=10)
            
            # Generate KML button
            def generate_kml():
                lat = earth_lat_entry.get()
                lon = earth_lon_entry.get()
                
                if not lat or not lon:
                    messagebox.showwarning("Missing Information", "Please enter both latitude and longitude to generate KML.")
                    return
                
                try:
                    lat_val = float(lat)
                    lon_val = float(lon)
                    
                    # Validate coordinates
                    if not (-90 <= lat_val <= 90) or not (-180 <= lon_val <= 180):
                        messagebox.showerror("Invalid Coordinates", "Latitude must be between -90 and 90, and longitude between -180 and 180.")
                        return
                    
                    # Generate KML content
                    kml_content = self.generate_kml_content(lat_val, lon_val)
                    
                    # Save to temporary file
                    temp_dir = tempfile.gettempdir()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    kml_file = os.path.join(temp_dir, f"classification_{timestamp}.kml")
                    
                    with open(kml_file, "w") as f:
                        f.write(kml_content)
                    
                    # Open the KML file
                    if os.name == 'nt':  # Windows
                        os.startfile(kml_file)
                    elif os.name == 'posix':  # macOS and Linux
                        webbrowser.open(f"file://{kml_file}")
                    
                    messagebox.showinfo("KML Generated", f"KML file saved to:\n{kml_file}")
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid numeric coordinates.")
                except Exception as e:
                    messagebox.showerror("KML Generation Error", f"Failed to generate KML: {str(e)}")
            
            kml_button = ctk.CTkButton(
                buttons_frame,
                text="📄 Generate KML",
                command=generate_kml,
                width=200,
                height=40,
                corner_radius=10,
                fg_color="#34A853",  # Google green
                hover_color="#2E8B47"
            )
            kml_button.pack(side="left", padx=10)
            
            # Static map preview
            map_frame = ctk.CTkFrame(earth_frame, corner_radius=10)
            map_frame.pack(fill="both", expand=True, pady=10, padx=10)
            
            map_title = ctk.CTkLabel(
                map_frame,
                text="📍 Location Preview",
                font=("Arial", 16, "bold")
            )
            map_title.pack(pady=(10, 5))
            
            # Map preview canvas
            map_canvas = tk.Canvas(map_frame, bg="#2b2b2b", highlightthickness=0)
            map_canvas.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Function to update map preview
            def update_map_preview(event=None):
                lat = earth_lat_entry.get()
                lon = earth_lon_entry.get()
                
                if not lat or not lon:
                    map_canvas.delete("all")
                    map_canvas.create_text(
                        map_canvas.winfo_width() // 2,
                        map_canvas.winfo_height() // 2,
                        text="Enter coordinates to preview location",
                        fill="white",
                        font=("Arial", 12)
                    )
                    return
                
                try:
                    lat_val = float(lat)
                    lon_val = float(lon)
                    
                    # Validate coordinates
                    if not (-90 <= lat_val <= 90) or not (-180 <= lon_val <= 180):
                        return
                    
                    # Use Esri World Imagery for preview
                    # Calculate tile coordinates
                    zoom_val = 15  # Fixed zoom for preview
                    n = 2.0 ** zoom_val
                    xtile = int((lon_val + 180.0) / 360.0 * n)
                    ytile = int((1.0 - math.log(math.tan(math.radians(lat_val)) + 1.0 / math.cos(math.radians(lat_val))) / math.pi) / 2.0 * n)
                    
                    # Get Esri World Imagery tile
                    tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_val}/{ytile}/{xtile}"
                    
                    # Fetch the tile
                    response = requests.get(tile_url)
                    img_data = response.content
                    
                    # Open the tile image
                    tile_img = Image.open(io.BytesIO(img_data))
                    
                    # Scale up the tile to fill the canvas
                    canvas_width = map_canvas.winfo_width()
                    canvas_height = map_canvas.winfo_height()
                    if canvas_width > 1 and canvas_height > 1:
                        scaled_img = tile_img.resize((canvas_width, canvas_height), Image.LANCZOS)
                        
                        # Display the image
                        map_canvas.esri_photo = ImageTk.PhotoImage(scaled_img)
                        map_canvas.delete("all")
                        map_canvas.create_image(0, 0, anchor="nw", image=map_canvas.esri_photo)
                        
                        # Add provider text
                        map_canvas.create_text(
                            10,
                            10,
                            text=f"Provider: Esri World Imagery",
                            fill="white",
                            font=("Arial", 10, "bold"),
                            anchor="nw"
                        )
                except Exception as e:
                    map_canvas.delete("all")
                    map_canvas.create_text(
                        map_canvas.winfo_width() // 2,
                        map_canvas.winfo_height() // 2,
                        text=f"Error loading map: {str(e)}",
                        fill="white",
                        font=("Arial", 12)
                    )
            
            # Bind entry changes to update map preview
            earth_lat_entry.bind("<KeyRelease>", update_map_preview)
            earth_lon_entry.bind("<KeyRelease>", update_map_preview)
            
            # Initial map preview
            update_map_preview()
            
            # Function to properly close the window
            def close_data_window():
                # Release the grab before destroying the window
                data_window.grab_release()
                data_window.destroy()
            
            # Handle window close event
            data_window.protocol("WM_DELETE_WINDOW", close_data_window)
            
            # Close button
            close_btn = ctk.CTkButton(
                data_window,
                text="Close",
                command=close_data_window,
                width=120,
                height=40,
                corner_radius=10,
                fg_color="#F44336",
                hover_color="#D32F2F"
            )
            close_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Image Data Error", f"Failed to display image data: {str(e)}")

    def show_analyse_window(self):
        """Show a window to analyze 4 images one by one"""
        # Clear any existing quadrant windows
        for window in self.quadrant_windows:
            if window.winfo_exists():
                window.destroy()
        self.quadrant_windows = []
        
        # Start with the first quadrant
        quadrant_window = QuadrantAnalysisWindow(self, self, 1)
        self.wait_window(quadrant_window)

    def fetch_satellite_image(self, lat_entry, lon_entry, zoom_entry, parent_window=None, canvas=None):
        """Fetch and display the current satellite image for the given coordinates using Esri World Imagery"""
        lat = lat_entry.get()
        lon = lon_entry.get()
        zoom = zoom_entry.get()

        if not lat or not lon:
            messagebox.showwarning("Missing Information", "Please enter both latitude and longitude.")
            return

        try:
            lat_val = float(lat)
            lon_val = float(lon)
            
            # Validate coordinates
            if not (-90 <= lat_val <= 90) or not (-180 <= lon_val <= 180):
                messagebox.showerror("Invalid Coordinates", "Latitude must be between -90 and 90, and longitude between -180 and 180.")
                return
            
            # Set default zoom if not provided
            if not zoom:
                zoom = "15"
            try:
                zoom_val = int(zoom)
                if not (1 <= zoom_val <= 20):
                    zoom_val = 15
            except ValueError:
                zoom_val = 15

            # Create a new window for displaying the satellite image
            image_window = ctk.CTkToplevel(self)
            image_window.title(f"🛰️ Satellite Image - {lat_val}, {lon_val}")
            
            # Set window to almost full screen
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            image_window.geometry(f"{int(screen_width * 0.9)}x{int(screen_height * 0.9)}+{int(screen_width * 0.05)}+{int(screen_height * 0.05)}")
            
            # Make window modal
            image_window.transient(self)
            image_window.grab_set()
            image_window.focus_set()
            
            # Center the window
            image_window.update_idletasks()
            width = image_window.winfo_width()
            height = image_window.winfo_height()
            x = (image_window.winfo_screenwidth() // 2) - (width // 2)
            y = (image_window.winfo_screenheight() // 2) - (height // 2)
            image_window.geometry(f'{width}x{height}+{x}+{y}')
            
            # Store reference to the satellite image window
            self.satellite_image_window = image_window
            
            # Create a frame for the image
            image_frame = ctk.CTkFrame(image_window, fg_color="transparent")
            image_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create a canvas for the image
            image_canvas = tk.Canvas(image_frame, bg="#2b2b2b", highlightthickness=0)
            image_canvas.pack(fill="both", expand=True)
            
            # Add loading text
            loading_text = image_canvas.create_text(
                image_canvas.winfo_width() // 2,
                image_canvas.winfo_height() // 2,
                text="Loading satellite image...",
                fill="white",
                font=("Arial", 14)
            )
            
            # Update the window to ensure the loading text is displayed
            image_window.update()
            
            # Calculate tile coordinates for Esri World Imagery
            n = 2.0 ** zoom_val
            xtile = int((lon_val + 180.0) / 360.0 * n)
            ytile = int((1.0 - math.log(math.tan(math.radians(lat_val)) + 1.0 / math.cos(math.radians(lat_val))) / math.pi) / 2.0 * n)
            
            # Get Esri World Imagery tile
            tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_val}/{ytile}/{xtile}"
            
            # For a larger image, we'll fetch multiple tiles and stitch them together
            # Calculate how many tiles we need based on the window size
            canvas_width = image_canvas.winfo_width()
            canvas_height = image_canvas.winfo_height()
            
            # Tile size is 256x256
            tile_size = 256
            
            # Calculate how many tiles we need in each direction
            tiles_x = math.ceil(canvas_width / tile_size) + 2  # +2 for padding
            tiles_y = math.ceil(canvas_height / tile_size) + 2  # +2 for padding
            
            # Calculate the starting tile coordinates
            start_x = max(0, xtile - tiles_x // 2)
            start_y = max(0, ytile - tiles_y // 2)
            
            # Create a new image to stitch the tiles
            stitched_image = Image.new('RGB', (tiles_x * tile_size, tiles_y * tile_size))
            
            # Fetch and stitch the tiles
            for y in range(tiles_y):
                for x in range(tiles_x):
                    tile_x = start_x + x
                    tile_y = start_y + y
                    
                    # Calculate the maximum tile index for this zoom level
                    max_tile = 2 ** zoom_val - 1
                    
                    # Skip if the tile coordinates are out of bounds
                    if tile_x < 0 or tile_y < 0 or tile_x > max_tile or tile_y > max_tile:
                        continue
                    
                    try:
                        # Get the tile URL
                        tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_val}/{tile_y}/{tile_x}"
                        
                        # Fetch the tile
                        response = requests.get(tile_url)
                        img_data = response.content
                        
                        # Open the tile image
                        tile_img = Image.open(io.BytesIO(img_data))
                        
                        # Paste the tile into the stitched image
                        stitched_image.paste(tile_img, (x * tile_size, y * tile_size))
                    except Exception as e:
                        logger.warning(f"Error fetching tile {tile_x}, {tile_y}: {str(e)}")
                        continue
            
            # Calculate the center of the requested location in the stitched image
            center_x = (xtile - start_x) * tile_size + tile_size // 2
            center_y = (ytile - start_y) * tile_size + tile_size // 2
            
            # Calculate the crop area to center the requested location
            crop_left = max(0, center_x - canvas_width // 2)
            crop_top = max(0, center_y - canvas_height // 2)
            crop_right = min(stitched_image.width, crop_left + canvas_width)
            crop_bottom = min(stitched_image.height, crop_top + canvas_height)
            
            # Crop the image
            cropped_image = stitched_image.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            # Resize the image to fit the canvas
            resized_image = cropped_image.resize((canvas_width, canvas_height), Image.LANCZOS)
            
            # Store the PIL image for later processing
            image_window.pil_image = resized_image
            image_window.location_info = f"{lat_val}, {lon_val}"
            
            # Display the image
            image_canvas.satellite_photo = ImageTk.PhotoImage(resized_image)
            image_canvas.delete("all")
            image_canvas.create_image(0, 0, anchor="nw", image=image_canvas.satellite_photo)
            
            # Add location info text
            image_canvas.create_text(
                10,
                10,
                text=f"Location: {lat_val}, {lon_val} | Provider: Esri World Imagery | Zoom: {zoom_val}",
                fill="white",
                font=("Arial", 12, "bold"),
                anchor="nw"
            )
            
            # Add image size label
            size_label = ctk.CTkLabel(
                image_frame,
                text=f"Image Size: {resized_image.width} × {resized_image.height} pixels",
                font=("Arial", 12),
                text_color=("#B3E5FC", "#64B5F6")
            )
            size_label.pack(side="bottom", pady=5)
            
            # Create a button frame at the top
            button_frame = ctk.CTkFrame(image_window, fg_color=("gray20", "#37474F"), corner_radius=0)
            button_frame.pack(side="top", fill="x")
            
            # Add buttons to the button frame
            button_content = ctk.CTkFrame(button_frame, fg_color="transparent")
            button_content.pack(fill="both", padx=10, pady=5)
            
            # Add a close button
            close_button = ctk.CTkButton(
                button_content,
                text="✕ Close",
                command=image_window.destroy,
                width=100,
                height=35,
                corner_radius=8,
                fg_color="#F44336",
                hover_color="#D32F2F"
            )
            close_button.pack(side="right", padx=5)
            
            # Add a save quadrants button
            save_quadrants_button = ctk.CTkButton(
                button_content,
                text="💾 Save Quadrants",
                command=lambda: self.save_satellite_quadrants(image_window),
                width=150,
                height=35,
                corner_radius=8,
                fg_color="#388E3C",
                hover_color="#2E7D32"
            )
            save_quadrants_button.pack(side="right", padx=5)
            
            # Add a process quadrants button
            process_quadrants_button = ctk.CTkButton(
                button_content,
                text="🔍 Process Quadrants",
                command=lambda: self.process_fetched_image(image_window),
                width=150,
                height=35,
                corner_radius=8,
                fg_color="#FF9800",
                hover_color="#F57C00"
            )
            process_quadrants_button.pack(side="right", padx=5)
            
            # Add a title to the button frame
            title_label = ctk.CTkLabel(
                button_content,
                text=f"🛰️ Satellite Image - {lat_val}, {lon_val}",
                font=("Arial", 14, "bold"),
                text_color="white"
            )
            title_label.pack(side="left", padx=5)
            
        except Exception as e:
            if canvas:
                canvas.delete("all")
                canvas.create_text(
                    canvas.winfo_width() // 2,
                    canvas.winfo_height() // 2,
                    text=f"Error loading map: {str(e)}",
                    fill="white",
                    font=("Arial", 12)
                )
            else:
                messagebox.showerror("Error", f"Failed to load satellite image: {str(e)}")

    def save_satellite_quadrants(self, image_window):
        """Save the satellite image as 4 quadrants to the project folder"""
        if not hasattr(image_window, 'pil_image'):
            messagebox.showerror("Error", "No satellite image available.")
            return
        
        try:
            # Get the PIL image from the image window
            pil_image = image_window.pil_image
            location_info = getattr(image_window, 'location_info', "Unknown location")
            
            # Get image dimensions
            width, height = pil_image.size
            
            # Calculate quadrant dimensions
            quad_width = width // 2
            quad_height = height // 2
            
            # Define the four quadrants
            quadrants = [
                (0, 0, quad_width, quad_height),  # Top-left
                (quad_width, 0, width, quad_height),  # Top-right
                (0, quad_height, quad_width, height),  # Bottom-left
                (quad_width, quad_height, width, height)  # Bottom-right
            ]
            
            # Get the project directory
            project_dir = os.getcwd()
            
            # Create a "quadrants" directory if it doesn't exist
            quadrants_dir = os.path.join(project_dir, "quadrants")
            if not os.path.exists(quadrants_dir):
                os.makedirs(quadrants_dir)
            
            # Create a timestamp for the filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Process each quadrant
            for i, (x1, y1, x2, y2) in enumerate(quadrants):
                # Extract the quadrant as a PIL Image
                quadrant = pil_image.crop((x1, y1, x2, y2))
                
                # Save the quadrant to the quadrants directory
                filename = f"satellite_quadrant_{i+1}_{timestamp}.png"
                filepath = os.path.join(quadrants_dir, filename)
                quadrant.save(filepath)
            
            # Show a success message
            messagebox.showinfo("Success", f"Satellite image saved as 4 quadrants to:\n{quadrants_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save satellite quadrants: {str(e)}")

    def process_fetched_image(self, image_window):
        """Process the fetched satellite image into 4 parts and display classification for each"""
        if not hasattr(image_window, 'pil_image'):
            messagebox.showerror("Error", "No image available for processing.")
            return
        
        try:
            # Get the PIL image from the image window
            pil_image = image_window.pil_image
            location_info = getattr(image_window, 'location_info', "Unknown location")
            
            # Get image dimensions
            width, height = pil_image.size
            
            # Calculate quadrant dimensions
            quad_width = width // 2
            quad_height = height // 2
            
            # Define the four quadrants
            quadrants = [
                (0, 0, quad_width, quad_height),  # Top-left
                (quad_width, 0, width, quad_height),  # Top-right
                (0, quad_height, quad_width, height),  # Bottom-left
                (quad_width, quad_height, width, height)  # Bottom-right
            ]
            
            # Clear any existing quadrant windows
            for window in self.quadrant_windows:
                if window.winfo_exists():
                    window.destroy()
            self.quadrant_windows = []
            
            # Create a single window to display all quadrants
            quadrants_window = ctk.CTkToplevel(self)
            quadrants_window.title("🛰️ Satellite Image Quadrants Classification")
            quadrants_window.geometry("1400x900")
            
            # Make window modal
            quadrants_window.transient(self)
            quadrants_window.grab_set()
            quadrants_window.focus_set()
            
            # Center the window
            quadrants_window.update_idletasks()
            width = quadrants_window.winfo_width()
            height = quadrants_window.winfo_height()
            x = (quadrants_window.winfo_screenwidth() // 2) - (width // 2)
            y = (quadrants_window.winfo_screenheight() // 2) - (height // 2)
            quadrants_window.geometry(f'{width}x{height}+{x}+{y}')
            
            # Add to quadrant windows list
            self.quadrant_windows.append(quadrants_window)
            
            # Header with title and close button
            header = ctk.CTkFrame(quadrants_window, height=70, corner_radius=0,
                                 fg_color=("#0D47A1", "#1A237E"))
            header.pack(fill="x", padx=0, pady=0)
            
            header_content = ctk.CTkFrame(header, fg_color="transparent")
            header_content.pack(fill="both", padx=30, pady=10)

            ctk.CTkLabel(header_content,
                        text="🛰️ Satellite Image Quadrants Classification",
                        font=("Arial", 20, "bold"),
                        text_color="white").pack(side="left")

            # Close button
            close_button = ctk.CTkButton(
                header_content,
                text="✕ Close All",
                command=self.close_all_quadrant_windows,
                width=120,
                height=35,
                fg_color="#F44336",
                hover_color="#D32F2F"
            )
            close_button.pack(side="right")
            
            # Location info
            if location_info:
                location_label = ctk.CTkLabel(
                    quadrants_window,
                    text=f"📍 Location: {location_info}",
                    font=("Arial", 12),
                    text_color=("#1976D2", "#4FC3F7")
                )
                location_label.pack(pady=(10, 5))
            
            # Main content area
            main_content = ctk.CTkFrame(quadrants_window, fg_color="transparent")
            main_content.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Create a grid for the quadrants (2x2)
            grid_frame = ctk.CTkFrame(main_content, fg_color="transparent")
            grid_frame.pack(fill="both", expand=True)
            
            # Process each quadrant
            for i, (x1, y1, x2, y2) in enumerate(quadrants):
                # Extract the quadrant as a PIL Image
                quadrant = pil_image.crop((x1, y1, x2, y2))
                
                # Process the quadrant
                try:
                    # Save the quadrant to a temporary file
                    temp_dir = tempfile.gettempdir()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_file = os.path.join(temp_dir, f"quadrant_{i+1}_{timestamp}.png")
                    quadrant.save(temp_file)
                    
                    # Process the quadrant using the classifier
                    orig, colored = self.classifier.process_image(temp_file)
                    
                    # Convert to colored image
                    colored_quadrant = (colored * 255).astype(np.uint8)
                    
                    # Create a frame for this quadrant
                    row = i // 2
                    col = i % 2
                    quadrant_frame = ctk.CTkFrame(grid_frame, corner_radius=15)
                    quadrant_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
                    
                    # Configure grid weights
                    grid_frame.grid_rowconfigure(row, weight=1)
                    grid_frame.grid_columnconfigure(col, weight=1)
                    
                    # Quadrant title
                    title_label = ctk.CTkLabel(
                        quadrant_frame,
                        text=f"Quadrant {i+1}",
                        font=("Arial", 16, "bold")
                    )
                    title_label.pack(pady=(10, 5))
                    
                    # Images container
                    images_container = ctk.CTkFrame(quadrant_frame, fg_color="transparent")
                    images_container.pack(fill="both", expand=True, padx=10, pady=10)
                    
                    # Original image
                    orig_frame = ctk.CTkFrame(images_container, corner_radius=10)
                    orig_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
                    
                    orig_title = ctk.CTkLabel(
                        orig_frame,
                        text="Original",
                        font=("Arial", 12)
                    )
                    orig_title.pack(pady=(5, 5))
                    
                    # Resize for display
                    display_orig = ImageOps.contain(quadrant, (300, 200))
                    
                    orig_canvas = tk.Canvas(orig_frame, bg="#2b2b2b", highlightthickness=0, width=300, height=200)
                    orig_canvas.pack(padx=5, pady=5)
                    
                    orig_canvas.orig_photo = ImageTk.PhotoImage(display_orig)
                    orig_canvas.create_image(0, 0, anchor="nw", image=orig_canvas.orig_photo)
                    
                    # Classified image
                    class_frame = ctk.CTkFrame(images_container, corner_radius=10)
                    class_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
                    
                    class_title = ctk.CTkLabel(
                        class_frame,
                        text="Classified",
                        font=("Arial", 12)
                    )
                    class_title.pack(pady=(5, 5))
                    
                    # Resize for display
                    display_class = ImageOps.contain(Image.fromarray(colored_quadrant), (300, 200))
                    
                    class_canvas = tk.Canvas(class_frame, bg="#2b2b2b", highlightthickness=0, width=300, height=200)
                    class_canvas.pack(padx=5, pady=5)
                    
                    class_canvas.class_photo = ImageTk.PhotoImage(display_class)
                    class_canvas.create_image(0, 0, anchor="nw", image=class_canvas.class_photo)
                    
                    # Add a button to view full size
                    view_btn = ctk.CTkButton(
                        quadrant_frame,
                        text="🔍 View Full Size",
                        command=lambda idx=i, quad=quadrant, col_quad=colored_quadrant: self.view_quadrant_full_size(idx+1, quad, col_quad, location_info),
                        width=150,
                        height=30,
                        corner_radius=10
                    )
                    view_btn.pack(pady=10)
                    
                    # Clean up the temporary file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
                except Exception as e:
                    logger.error(f"Error processing quadrant {i+1}: {str(e)}")
                    messagebox.showerror("Processing Error", f"Failed to process quadrant {i+1}: {str(e)}")
            
            # Status bar
            status_var = ctk.StringVar(value="✅ All quadrants processed successfully")
            status_bar = ctk.CTkLabel(quadrants_window,
                                     textvariable=status_var,
                                     font=("Arial", 12),
                                     corner_radius=0,
                                     fg_color=("gray85", "gray25"))
            status_bar.pack(side="bottom", fill="x")
            
            # Show a success message
            messagebox.showinfo("Processing Complete", "The satellite image has been processed into 4 quadrants. All quadrants are displayed in a single window.")
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to process the satellite image: {str(e)}")

    def view_quadrant_full_size(self, quadrant_num, original_img, processed_img, location_info):
        """View a single quadrant in full size"""
        quadrant_window = QuadrantDisplayWindow(self, self, quadrant_num, original_img, Image.fromarray(processed_img), location_info)
        self.quadrant_windows.append(quadrant_window)

    def close_all_quadrant_windows(self):
        """Close all quadrant windows and return to the main dashboard"""
        # Close all quadrant windows
        for window in self.quadrant_windows:
            if window.winfo_exists():
                window.destroy()
        
        # Clear the list
        self.quadrant_windows = []
        
        # Close the parent satellite image window if it exists
        if hasattr(self, 'satellite_image_window') and self.satellite_image_window.winfo_exists():
            self.satellite_image_window.destroy()
        
        # Show the main dashboard
        self.show_main_application()

    def generate_kml_content(self, lat, lon):
        """Generate KML content for the classification results"""
        kml_template = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Satellite Image Classification</name>
    <description>Land cover classification results</description>
    <Style id="style_0">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/1.png</href>
        </Icon>
      </IconStyle>
    </Style>
    <Placemark>
      <name>Classification Location</name>
      <description>
        <![CDATA[
        <h3>Satellite Image Classification</h3>
        <p><b>Location:</b> {lat}, {lon}</p>
        <p><b>Date:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><b>Image:</b> {self.current_image_path.name if self.current_image_path else "Unknown"}</p>
        <h4>Classification Results:</h4>
        <ul>
        '''
        
        if self.current_counts:
            for class_name, count in self.current_counts.items():
                try:
                    class_idx = self.classifier.class_names.index(class_name)
                    color_hex = f"#{self.classifier.class_colors[class_idx][0]:02x}{self.classifier.class_colors[class_idx][1]:02x}{self.classifier.class_colors[class_idx][2]:02x}"
                except (ValueError, AttributeError, KeyError):
                    color_hex = "#808080"
                kml_template += f'          <li><span style="color:{color_hex}">■</span> {class_name}: {count} grids</li>\n'
        
        kml_template += f'''        </ul>
        ]]>
      </description>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>'''
        
        return kml_template

def main():
    app = ModernSatelliteApp()
    app.mainloop()

if __name__ == "__main__":
    main()