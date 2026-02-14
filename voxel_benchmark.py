from src import load_obj



def main():
    print("Hello")
    path = "obj/teapot.obj"
    load_obj(path=path)
    print("Object has been loaded")
    

if __name__ == "__main__":
    main()