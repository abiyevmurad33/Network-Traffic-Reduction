import vizdoom as vzd

def main():
    game = vzd.DoomGame()
    # We won't load a config yet. This test checks basic object construction.
    print("DoomGame object created:", type(game).__name__)
    print("OK: ViZDoom basic init path reached (config/scenario next step).")

if __name__ == "__main__":
    main()
