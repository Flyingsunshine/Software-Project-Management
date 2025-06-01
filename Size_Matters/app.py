import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

# Expanded database of recommendations with height considerations
RECOMMENDATIONS = {
    "slim": {
        "short": {  # Under 5'6" / 168cm
            "T-shirt": {
                "size": "S",
                "style": "Fitted t-shirts in petite sizes. Look for shorter length options."
            },
            "Trousers": {
                "size": "28-30 (Short/Petite)",
                "style": "Slim or straight leg with no break at the ankle. Avoid excessive cuffing."
            },
            "Jeans": {
                "size": "28-30 (Short/Petite)",
                "style": "Slim or straight cut jeans in short/petite length."
            },
            "Dress Shirt": {
                "size": "14-15 (Short)",
                "style": "Slim-fit dress shirts with shorter sleeves and length."
            }
        },
        "average": {  # 5'6" to 6'0" / 168-183cm
            "T-shirt": {
                "size": "M",
                "style": "Fitted or slim-fit t-shirts work best for a slim body. Try layering or patterns to add volume."
            },
            "Trousers": {
                "size": "30-32 (Regular)",
                "style": "Straight or slim-fit trousers. Avoid baggy styles."
            },
            "Jeans": {
                "size": "30-32 (Regular)",
                "style": "Slim or straight cut jeans. Consider tapered styles to add definition."
            },
            "Dress Shirt": {
                "size": "15-15.5 (Regular)",
                "style": "Slim-fit dress shirts. Tuck in for a cleaner look."
            }
        },
        "tall": {  # Over 6'0" / 183cm
            "T-shirt": {
                "size": "L (Tall)",
                "style": "Look for tall sizes with extra length in body and sleeves."
            },
            "Trousers": {
                "size": "32-34 (Long)",
                "style": "Slim or straight leg in tall/long length. Ensure inseam is sufficient."
            },
            "Jeans": {
                "size": "32-34 (Long/Tall)",
                "style": "Slim or straight cut jeans with at least 34\" inseam."
            },
            "Dress Shirt": {
                "size": "15.5-16 (Tall)",
                "style": "Slim-fit dress shirts with extra sleeve length. Look for tall sizes."
            }
        }
    },
    "pot": {
        "short": {  # Under 5'6" / 168cm
            "T-shirt": {
                "size": "L",
                "style": "Relaxed-fit t-shirts that don't cling to the midsection. Petite length to avoid excess fabric."
            },
            "Trousers": {
                "size": "36-38 (Short)",
                "style": "Mid-rise, straight trousers with comfort waist in short length."
            },
            "Jeans": {
                "size": "36-38 (Short)",
                "style": "Straight-leg jeans with mid-rise in petite/short length."
            },
            "Dress Shirt": {
                "size": "16.5-17 (Short)",
                "style": "Regular fit dress shirts with room in the midsection. Short length to avoid excess fabric."
            }
        },
        "average": {  # 5'6" to 6'0" / 168-183cm
            "T-shirt": {
                "size": "XL",
                "style": "Go for relaxed-fit or A-line t-shirts. V-necks and darker colors are flattering."
            },
            "Trousers": {
                "size": "38-40 (Regular)",
                "style": "Mid-rise, straight or slightly tapered trousers with elastic waistband for comfort."
            },
            "Jeans": {
                "size": "38-40 (Regular)",
                "style": "Straight-leg jeans with mid-rise. Avoid skinny jeans and low-rise styles."
            },
            "Dress Shirt": {
                "size": "17-17.5 (Regular)",
                "style": "Regular or classic fit dress shirts. Avoid slim-fit styles that might pull at the buttons."
            }
        },
        "tall": {  # Over 6'0" / 183cm
            "T-shirt": {
                "size": "XL (Tall)",
                "style": "Relaxed-fit t-shirts in tall sizes for extra length. V-necks elongate the torso."
            },
            "Trousers": {
                "size": "38-40 (Long)",
                "style": "Mid-rise, straight trousers with long inseam. Look for comfort waistband."
            },
            "Jeans": {
                "size": "38-40 (Long)",
                "style": "Straight-leg jeans with at least 34\" inseam. Avoid low-rise styles."
            },
            "Dress Shirt": {
                "size": "17-18 (Tall)",
                "style": "Regular fit dress shirts with extra sleeve and body length. Tall sizes recommended."
            }
        }
    },
    "hourglass": {
        "short": {  # Under 5'6" / 168cm
            "T-shirt": {
                "size": "S-M",
                "style": "Fitted styles in petite sizes that highlight the waist. Avoid boxy cuts."
            },
            "Trousers": {
                "size": "30-32 (Short)",
                "style": "High-waisted, straight trousers in petite length to emphasize your waist."
            },
            "Jeans": {
                "size": "30-32 (Short)",
                "style": "Mid or high-rise jeans with stretch in petite length. Curvy fit recommended."
            },
            "Dress Shirt": {
                "size": "14-15 (Short)",
                "style": "Tailored shirts with darts in petite sizes that follow your curves."
            }
        },
        "average": {  # 5'6" to 6'0" / 168-183cm
            "T-shirt": {
                "size": "M-L",
                "style": "Fitted styles that highlight the waist. Wrap tops and peplum styles are flattering."
            },
            "Trousers": {
                "size": "32-34 (Regular)",
                "style": "High-waisted, straight or slightly bootcut trousers that emphasize your waist."
            },
            "Jeans": {
                "size": "32-34 (Regular)",
                "style": "Mid or high-rise jeans with stretch. Straight or bootcut styles balance proportions."
            },
            "Dress Shirt": {
                "size": "16-16.5 (Regular)",
                "style": "Tailored shirts with darts or princess seams that follow your curves."
            }
        },
        "tall": {  # Over 6'0" / 183cm
            "T-shirt": {
                "size": "L (Tall)",
                "style": "Fitted styles in tall sizes that highlight the waist. Extra length prevents riding up."
            },
            "Trousers": {
                "size": "34-36 (Long)",
                "style": "High-waisted, straight or bootcut trousers with sufficient inseam length."
            },
            "Jeans": {
                "size": "34-36 (Long)",
                "style": "Mid or high-rise jeans with at least 34\" inseam. Look for curvy tall fits."
            },
            "Dress Shirt": {
                "size": "16.5-17 (Tall)",
                "style": "Tailored shirts with extra length. Look for tall sizes with defined waist."
            }
        }
    },
    "inverted_triangle": {
        "short": {  # Under 5'6" / 168cm
            "T-shirt": {
                "size": "M",
                "style": "A-line t-shirts in petite sizes. V-necks draw attention away from shoulders."
            },
            "Trousers": {
                "size": "32-34 (Short)",
                "style": "Wide-leg or bootcut trousers in petite length to balance proportions."
            },
            "Jeans": {
                "size": "32-34 (Short)",
                "style": "Bootcut or flared jeans in petite length with details on the pockets."
            },
            "Dress Shirt": {
                "size": "15-16 (Short)",
                "style": "Shirts with minimal shoulder details in petite sizes. Avoid excessive shoulder padding."
            }
        },
        "average": {  # 5'6" to 6'0" / 168-183cm
            "T-shirt": {
                "size": "L",
                "style": "A-line or flared t-shirts that balance broader shoulders. Avoid boat necks and cap sleeves."
            },
            "Trousers": {
                "size": "34-36 (Regular)",
                "style": "Wide-leg or bootcut trousers to balance proportions. Add details at the hips."
            },
            "Jeans": {
                "size": "34-36 (Regular)",
                "style": "Bootcut or flared jeans with details on the pockets to add volume to hips."
            },
            "Dress Shirt": {
                "size": "16-17 (Regular)",
                "style": "Shirts with minimal shoulder details. V-necks and open collars work well."
            }
        },
        "tall": {  # Over 6'0" / 183cm
            "T-shirt": {
                "size": "L-XL (Tall)",
                "style": "A-line t-shirts in tall sizes. V-necks and scoop necks minimize broad shoulders."
            },
            "Trousers": {
                "size": "36-38 (Long)",
                "style": "Wide-leg or bootcut trousers with long inseam to balance proportions."
            },
            "Jeans": {
                "size": "36-38 (Long)",
                "style": "Bootcut or flared jeans with at least 34\" inseam. Look for designs with hip details."
            },
            "Dress Shirt": {
                "size": "17-18 (Tall)",
                "style": "Shirts with extra length and minimal shoulder details. Tall sizes recommended."
            }
        }
    }
}

# Display names for body types
BODY_TYPE_LABELS = {
    "slim": "Slim/Rectangle",
    "pot": "Fuller/Apple",
    "hourglass": "Hourglass",
    "inverted_triangle": "Inverted Triangle"
}

class SizeMattersApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Size Matters - Find Your Perfect Fit")
        self.geometry("800x750")
        self.configure(bg="#f5f5f5")
        
        # Initialize variables
        self.selected_body_type = None
        self.height_category = None
        self.height_value = None
        self.height_unit = tk.StringVar(value="cm")

        # Create images directory if it doesn't exist
        if not os.path.exists("images"):
            os.makedirs("images")
            print("Please place your body type images in the 'images' folder")

        # Scrollable frame setup
        main_frame = tk.Frame(self)
        main_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(main_frame, bg="#f5f5f5")
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        self.scroll_frame = tk.Frame(canvas, bg="#f5f5f5")
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header
        header_frame = tk.Frame(self.scroll_frame, bg="#3a7ca5", pady=15)
        header_frame.pack(fill="x")
        
        tk.Label(header_frame, text="Size Matters", font=("Arial", 22, "bold"), 
                 bg="#3a7ca5", fg="white").pack()
        tk.Label(header_frame, text="Find your perfect fit without measurements", 
                 font=("Arial", 12), bg="#3a7ca5", fg="white").pack()

        # Main content area
        self.content_frame = tk.Frame(self.scroll_frame, bg="#f5f5f5")
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Show body type selection screen
        self.show_body_type_selection()

    def show_body_type_selection(self):
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        # Instructions
        tk.Label(self.content_frame, text="Step 1: Select the image that best matches your body type",
                 font=("Arial", 14, "bold"), bg="#f5f5f5", pady=10).pack()

        # Image buttons
        img_frame = tk.Frame(self.content_frame, bg="#f5f5f5")
        img_frame.pack()

        self.images = {}
        
        # Try to load images
        try:
            # First row of images
            row1_frame = tk.Frame(img_frame, bg="#f5f5f5")
            row1_frame.pack(pady=10)
            
            # Second row of images
            row2_frame = tk.Frame(img_frame, bg="#f5f5f5")
            row2_frame.pack(pady=10)
            
            # Define which body types go in which row
            row1_types = ["slim", "pot"]
            row2_types = ["hourglass", "inverted_triangle"]
            
            # Load first row
            for name in row1_types:
                self.load_body_type_image(name, row1_frame)
                
            # Load second row
            for name in row2_types:
                self.load_body_type_image(name, row2_frame)
                
        except Exception as e:
            print(f"Error loading images: {e}")

    def load_body_type_image(self, name, parent_frame):
        img_path = os.path.join("images", f"{name}.jpg")
        if os.path.exists(img_path):
            pil_img = Image.open(img_path).resize((150, 220))
            self.images[name] = ImageTk.PhotoImage(pil_img)
            
            # Create a frame for each image with label
            type_frame = tk.Frame(parent_frame, bg="#f5f5f5")
            type_frame.pack(side="left", padx=20)
            
            btn = tk.Button(type_frame, image=self.images[name], 
                            command=lambda n=name: self.select_body_type(n))
            btn.pack()
            
            # Add label below image
            tk.Label(type_frame, text=BODY_TYPE_LABELS[name], font=("Arial", 12), 
                     bg="#f5f5f5").pack(pady=5)
        else:
            print(f"Warning: {img_path} not found")
            placeholder = tk.Frame(parent_frame, width=150, height=220, bg="#dddddd")
            placeholder.pack(side="left", padx=20)
            tk.Label(placeholder, text=f"Place {name}.jpg\nin images folder", 
                     bg="#dddddd").place(relx=0.5, rely=0.5, anchor="center")

    def select_body_type(self, body_type):
        self.selected_body_type = body_type
        self.show_height_input()
        
    def show_height_input(self):
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        # Instructions
        tk.Label(self.content_frame, text=f"Step 2: Enter your height",
                 font=("Arial", 14, "bold"), bg="#f5f5f5", pady=10).pack()
                 
        # Selected body type info
        body_type_frame = tk.Frame(self.content_frame, bg="#e6f2ff", pady=10)
        body_type_frame.pack(fill="x", pady=10)
        
        tk.Label(body_type_frame, text=f"Selected Body Type: {BODY_TYPE_LABELS[self.selected_body_type]}", 
                 font=("Arial", 12, "bold"), bg="#e6f2ff").pack()
        
        # Height input frame
        height_frame = tk.Frame(self.content_frame, bg="#f5f5f5", pady=20)
        height_frame.pack()
        
        # Height unit selection
        unit_frame = tk.Frame(height_frame, bg="#f5f5f5")
        unit_frame.pack(pady=10)
        
        tk.Label(unit_frame, text="Select unit:", font=("Arial", 12), bg="#f5f5f5").pack(side="left", padx=5)
        
        cm_rb = tk.Radiobutton(unit_frame, text="Centimeters (cm)", variable=self.height_unit, 
                               value="cm", command=self.update_height_input, bg="#f5f5f5")
        cm_rb.pack(side="left", padx=10)
        
        feet_rb = tk.Radiobutton(unit_frame, text="Feet & Inches", variable=self.height_unit, 
                                value="feet", command=self.update_height_input, bg="#f5f5f5")
        feet_rb.pack(side="left", padx=10)
        
        # Frame for height input fields
        self.height_input_frame = tk.Frame(height_frame, bg="#f5f5f5", pady=10)
        self.height_input_frame.pack()
        
        # Show appropriate height input based on selected unit
        self.update_height_input()
        
        # Continue button
        continue_btn = tk.Button(self.content_frame, text="Get Size Recommendations", 
                               command=self.show_recommendations,
                               bg="#3a7ca5", fg="white", font=("Arial", 12, "bold"),
                               padx=15, pady=8)
        continue_btn.pack(pady=20)
        
        # Back button
        back_btn = tk.Button(self.content_frame, text="Back to Body Type Selection", 
                           command=self.show_body_type_selection,
                           font=("Arial", 10))
        back_btn.pack()
        
    def update_height_input(self):
        # Clear height input frame
        for widget in self.height_input_frame.winfo_children():
            widget.destroy()
            
        if self.height_unit.get() == "cm":
            # CM input
            tk.Label(self.height_input_frame, text="Height (cm):", 
                   font=("Arial", 12), bg="#f5f5f5").pack(side="left", padx=5)
            
            self.cm_entry = tk.Entry(self.height_input_frame, width=10, font=("Arial", 12))
            self.cm_entry.pack(side="left", padx=5)
            
            # Example
            tk.Label(self.height_input_frame, text="(e.g., 170)", 
                   font=("Arial", 10), bg="#f5f5f5").pack(side="left", padx=5)
            
        else:
            # Feet and inches input
            tk.Label(self.height_input_frame, text="Feet:", 
                   font=("Arial", 12), bg="#f5f5f5").pack(side="left", padx=5)
            
            self.feet_entry = tk.Entry(self.height_input_frame, width=5, font=("Arial", 12))
            self.feet_entry.pack(side="left", padx=5)
            
            tk.Label(self.height_input_frame, text="Inches:", 
                   font=("Arial", 12), bg="#f5f5f5").pack(side="left", padx=5)
            
            self.inches_entry = tk.Entry(self.height_input_frame, width=5, font=("Arial", 12))
            self.inches_entry.pack(side="left", padx=5)
            
            # Example
            tk.Label(self.height_input_frame, text="(e.g., 5 feet 8 inches)", 
                   font=("Arial", 10), bg="#f5f5f5").pack(side="left", padx=5)
    
    def determine_height_category(self):
        try:
            if self.height_unit.get() == "cm":
                height_cm = float(self.cm_entry.get())
                self.height_value = height_cm
            else:
                feet = float(self.feet_entry.get())
                inches = float(self.inches_entry.get())
                height_cm = (feet * 12 + inches) * 2.54
                self.height_value = height_cm
                
            # Determine height category
            if height_cm < 168:  # Under 5'6"
                return "short"
            elif height_cm > 183:  # Over 6'0"
                return "tall"
            else:  # 5'6" to 6'0"
                return "average"
                
        except ValueError:
            return None
    
    def format_height_display(self):
        if self.height_unit.get() == "cm":
            return f"{self.height_value:.1f} cm"
        else:
            total_inches = self.height_value / 2.54
            feet = int(total_inches // 12)
            inches = round(total_inches % 12)
            return f"{feet}'{inches}\""
            
    def show_recommendations(self):
        # Determine height category
        self.height_category = self.determine_height_category()
        
        if not self.height_category:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid height.")
            return
            
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        # Get recommendations based on body type and height
        rec = RECOMMENDATIONS[self.selected_body_type][self.height_category]
        
        # Results header
        header_frame = tk.Frame(self.content_frame, bg="#3a7ca5", pady=10)
        header_frame.pack(fill="x", pady=10)
        
        tk.Label(header_frame, text="Your Personalized Size Recommendations", 
                 font=("Arial", 14, "bold"), bg="#3a7ca5", fg="white").pack()
                 
        # Summary of selections
        summary_frame = tk.Frame(self.content_frame, bg="#e6f2ff", pady=15)
        summary_frame.pack(fill="x", pady=5)
        
        tk.Label(summary_frame, text=f"Body Type: {BODY_TYPE_LABELS[self.selected_body_type]}", 
                 font=("Arial", 12, "bold"), bg="#e6f2ff").pack(anchor="w", padx=20)
                 
        tk.Label(summary_frame, text=f"Height: {self.format_height_display()}", 
                 font=("Arial", 12, "bold"), bg="#e6f2ff").pack(anchor="w", padx=20)
        
        # Recommendations with alternating background colors
        bg_colors = ["#f0f8ff", "#ffffff"]
        i = 0
        
        for item, details in rec.items():
            item_frame = tk.Frame(self.content_frame, bg=bg_colors[i % 2], pady=15)
            item_frame.pack(fill="x", pady=2)
            
            tk.Label(item_frame, text=item, font=("Arial", 13, "bold"), 
                     bg=bg_colors[i % 2]).pack(anchor="w", padx=20)
            
            size_frame = tk.Frame(item_frame, bg=bg_colors[i % 2])
            size_frame.pack(fill="x", anchor="w", padx=30)
            
            tk.Label(size_frame, text="Recommended Size:", font=("Arial", 11, "bold"), 
                     bg=bg_colors[i % 2]).pack(side="left")
                     
            tk.Label(size_frame, text=details['size'], font=("Arial", 11), 
                     bg=bg_colors[i % 2]).pack(side="left", padx=5)
            
            style_label = tk.Label(item_frame, text=f"Style Tip: {details['style']}", 
                                 font=("Arial", 11), bg=bg_colors[i % 2], 
                                 wraplength=700, justify="left")
            style_label.pack(anchor="w", padx=30, pady=(5,0))
            
            i += 1

        # Footer note
        tk.Label(self.content_frame, 
                 text="These recommendations are general guidelines and may vary by brand.",
                 font=("Arial", 10, "italic"), bg="#f5f5f5").pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(self.content_frame, bg="#f5f5f5")
        button_frame.pack(pady=15)
        
        # Try different height button
        height_btn = tk.Button(button_frame, text="Try Different Height", 
                             command=lambda: self.show_height_input(),
                             bg="#3a7ca5", fg="white", font=("Arial", 11),
                             padx=10, pady=5)
        height_btn.pack(side="left", padx=10)
        
        # Try different body type button
        body_btn = tk.Button(button_frame, text="Try Different Body Type", 
                           command=self.show_body_type_selection,
                           bg="#3a7ca5", fg="white", font=("Arial", 11),
                           padx=10, pady=5)
        body_btn.pack(side="left", padx=10)

if __name__ == "__main__":
    app = SizeMattersApp()
    app.mainloop()
