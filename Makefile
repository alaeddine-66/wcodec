TARGET   = main
CC       = gcc
LD       = gcc
OBJ_DIR  = build
UNITY_DIR = tests/unity

CFLAGS  += -Wall -Wextra -std=c99 -Iinclude -I$(UNITY_DIR) -MMD -MP 
LDFLAGS  = -lm

URL_BASE    = https://raw.githubusercontent.com/ThrowTheSwitch/Unity/master/src
UNITY_FILES = $(UNITY_DIR)/unity.c $(UNITY_DIR)/unity.h $(UNITY_DIR)/unity_internals.h

SRC_FILES   = $(wildcard src/*.c)
OBJ_FILES   = $(patsubst src/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))
TEST_OBJS   = $(filter-out $(OBJ_DIR)/main.o, $(OBJ_FILES))

TEST_SRCS   = $(wildcard tests/test_*.c)
TEST_BINS   = $(TEST_SRCS:.c=.bin)
# Correction des dépendances pour inclure les fichiers .d des tests correctement
DEPS        = $(OBJ_FILES:.o=.d) $(patsubst tests/%.c,$(OBJ_DIR)/%.d,$(TEST_SRCS))

.PHONY: all debug tests clean

##@ compilation
all: CFLAGS  += -g -O2 -fsanitize=address --coverage $(ERR)
all: LDFLAGS += -g -fsanitize=address --coverage
all: $(TARGET) $(TEST_BINS) 

debug: CFLAGS  += -g -Og -fsanitize=address 
debug: LDFLAGS += -g -fsanitize=address 
debug: clean $(TARGET) $(TEST_BINS) 

# Inclusion des dépendances auto
-include $(DEPS)

# Création des dossiers à la volée
$(OBJ_DIR) $(UNITY_DIR):
	mkdir -p $@

# Téléchargement de Unity
$(UNITY_DIR)/%: | $(UNITY_DIR)
	@curl -Ls $(URL_BASE)/$(notdir $@) -o $@

# Compilation de Unity
$(OBJ_DIR)/unity.o: $(UNITY_FILES) | $(OBJ_DIR)
	$(CC) -c $(CFLAGS) $< -o $@

# Compilation des sources
$(OBJ_DIR)/%.o: src/%.c | $(OBJ_DIR)
	$(CC) -c $(CFLAGS) $< -o $@

# Compilation des objets de test (Correction du chemin du pattern)
$(OBJ_DIR)/%.o: tests/%.c | $(OBJ_DIR)
	$(CC) -c $(CFLAGS) $< -o $@

# Édition de liens pour les binaires de test (Correction ici : tests/%.bin)
tests/%.bin: $(OBJ_DIR)/%.o $(TEST_OBJS) $(OBJ_DIR)/unity.o
	$(LD) $^ $(LDFLAGS) -o $@

# Édition de liens pour le binaire principal
$(TARGET): $(OBJ_FILES) 
	$(LD) $^ $(LDFLAGS) -o $@

clean: 
	rm -rf $(TARGET) $(OBJ_DIR) tests/*.bin