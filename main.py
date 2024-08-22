from vision import *
def main():
    while True:

        #captura de imagens
        cod = valida_item()
        if cod == 1:
            print("O objeto depositado é do tipo vidro\n")
            #acionar servo motor 1

        elif cod == 2:
            print("O objeto depositado é do tipo metal\n")
            # acionar servo motor 2

        #inicio do jogo

        print("Muito bem! o jogo será iniciado em instantes...")
        break


if __name__ == "__main__":
    main()