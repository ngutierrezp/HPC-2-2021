


CC=gcc
CCFLAGS=-Wall
SRC=src
INCL=incl
EXECUTABLE_NAME=hough


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
.SILENT: all $(EXECUTABLE_NAME) $(SRC)/%.o clean $(CC) make $(SOURCES) $(OBJECTS)

all: clean $(EXECUTABLE_NAME)
	@echo "$(PUR_COLOR)Ejecutable generado!$(NO_COLOR) Nombre: $(OK_COLOR)$(EXECUTABLE_NAME)$(NO_COLOR) "

$(EXECUTABLE_NAME): $(OBJECTS)
	@echo "Generando ejecutable ..."
	$(CC) -o $@ $^ hough.c $(LDFLAGS) 

$(SRC)/%.o: $(SRC)/%.c $(INCL)/%.h
	@echo "Generando archivos object de $@ ...."
	($(CC) -lm -I$(SRC) -c $< -o $@ && echo "$(OK_COLOR)[OK]$(NO_COLOR)") \
		||  (echo "$(ERROR_COLOR)[ERROR]$(NO_COLOR)" && exit 1; )
clean:
	@echo "Eliminado $(WARN_COLOR).o $(NO_COLOR)y $(WARN_COLOR)$(EXECUTABLE_NAME) $(NO_COLOR) antiguos..."
	rm -f $(SRC)/*.o $(EXECUTABLE_NAME)

