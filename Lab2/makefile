


CC=gcc
CCFLAGS=-Wall -lm -fopenmp 
SRC=src
INCL=incl
EXECUTABLE_NAME=bomb
COMPILED=false


#### COLORS ####
NO_COLOR :=\033[0;0m

OK_COLOR :=\033[0;32m

ERROR_COLOR :=\033[0;31m

WARN_COLOR :=\033[0;36m

SUSF_PRINT :=\033[0;34m

PUR_COLOR := \033[0;35m


##################
# instancia de compilación


SOURCES := $(wildcard $(SRC)/*.c)
OBJECTS := $(patsubst $(SRC)/%.c, $(SRC)/%.o, $(SOURCES))
.SILENT: all  $(SRC)/%.o clean $(CC) make $(SOURCES) $(OBJECTS) $(EXECUTABLE_NAME).o
.PHONY: $(EXECUTABLE_NAME)

all: $(EXECUTABLE_NAME)
	@if [ $(COMPILED) = true ]; then \
		echo "$(PUR_COLOR)Ejecutable generado!$(NO_COLOR) Nombre: $(OK_COLOR)$(EXECUTABLE_NAME)$(NO_COLOR) ";\
	fi

$(EXECUTABLE_NAME): $(OBJECTS) $(EXECUTABLE_NAME).o
	@if [ $(COMPILED) = false ]; then \
		echo "$(OK_COLOR)**************************************************$(NO_COLOR)";\
		echo "$(OK_COLOR)*$(NO_COLOR)  No existen cambios, no se volverá a compilar  $(OK_COLOR)*$(NO_COLOR)";\
		echo "$(OK_COLOR)**************************************************$(NO_COLOR)";\
	else\
		echo "Generando ejecutable ...";\
		$(CC) -o $@ $^ $(CCFLAGS) -L./incl -lniceprint ;\
	fi

$(EXECUTABLE_NAME).o: $(EXECUTABLE_NAME).c
	@echo "Generando archivos object de $@ ...."
	($(CC)  -I$(SRC) -c $(EXECUTABLE_NAME).c  -o $@ $(CCFLAGS) && echo "$(OK_COLOR)[OK]$(NO_COLOR)") \
		||  (echo "$(ERROR_COLOR)[ERROR]$(NO_COLOR)" && exit 1; )
	$(eval COMPILED=true)
	

$(SRC)/%.o: $(SRC)/%.c $(INCL)/%.h
	@echo "Generando archivos object de $@ ...."
	($(CC)  -I$(SRC) -c $< -o $@ $(CCFLAGS) && echo "$(OK_COLOR)[OK]$(NO_COLOR)") \
		||  (echo "$(ERROR_COLOR)[ERROR]$(NO_COLOR)" && exit 1; )
	$(eval COMPILED=true)
clean:
	@echo "Eliminado $(WARN_COLOR).o $(NO_COLOR)y $(WARN_COLOR)$(EXECUTABLE_NAME) $(NO_COLOR) antiguos..."
	rm -f $(SRC)/*.o $(EXECUTABLE_NAME) *.o

