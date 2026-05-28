.PHONY: all clean debug

CC = gcc
CFLAGS = -Wall -ansi -pedantic -g -std=c99
LDFLAGS =

EXECUTABLE = main

SRC_DIR = src
OBJ_DIR = obj

SOURCE = $(wildcard $(SRC_DIR)/*.c)
OBJETS = $(SOURCE:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(EXECUTABLE)

debug: CFLAGS += -g
debug: all

$(EXECUTABLE): $(OBJETS)
	$(CC) $(CFLAGS) $(OBJETS) $(LDFLAGS) -o $(EXECUTABLE)

# compilation des .c de src vers obj
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -MMD -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) *.d $(EXECUTABLE)

-include $(OBJETS:.o=.d)
