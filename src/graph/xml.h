/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef XML_H_
#define XML_H_

// A few constraints to make the implementation easy
#define MAX_STR_LEN 32
#define MAX_ATTR_COUNT 8
#define MAX_SUBS 32
#define MAX_NODES 1024

#define KEY_TYPE_NONE 0
#define KEY_TYPE_INT 1
#define KEY_TYPE_STR 2

#define NODE_TYPE_NONE 0
#define NODE_TYPE_OPEN 1
#define NODE_TYPE_CLOSE 2
#define NODE_TYPE_SINGLE 3

struct xmlNode {
  char name[MAX_STR_LEN];
  struct {
    char key[MAX_STR_LEN];
    char strValue[MAX_STR_LEN];
    int intValue;
    int type;
  } attrs[MAX_ATTR_COUNT+1]; // Need an extra one to consume extra params
  int nAttrs;
  int type;
  struct xmlNode* parent;
  struct xmlNode* subs[MAX_SUBS];
  int nSubs;
};

struct xmlSystem {
  struct xmlNode nodes[MAX_NODES];
  int maxIndex;
};

/* File functions */
ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct xmlSystem* system);
ncclResult_t ncclTopoDumpSystemToXml(const char* xmlTopoFile, struct xmlSystem* system);

/* Auto-detect functions */
ncclResult_t ncclTopoFillGpu(struct xmlSystem* system, const char* busId, struct xmlNode** gpuNode);
ncclResult_t ncclTopoFillNic(struct xmlSystem* system, const char* sysPath, struct xmlNode** netNode);

/**************/
/* XML Struct */
/* Functions  */
/**************/

static ncclResult_t xmlGetAttrIndex(struct xmlNode* node, const char* attrName, int* index) {
  *index = -1;
  for (int a=0; a<node->nAttrs; a++) {
    if (strcmp(node->attrs[a].key, attrName) == 0) {
      *index = a;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrInt(struct xmlNode* node, const char* attrName, int* value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    WARN("Attribute %s of node %s not found\n", attrName, node->name);
    return ncclInternalError;
  }
  if (node->attrs[index].type != KEY_TYPE_INT) {
    WARN("Attribute %s of node %s is not an int (%d)\n", attrName, node->name, node->attrs[index].type);
    return ncclInternalError;
  }
  *value = node->attrs[index].intValue;
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrStr(struct xmlNode* node, const char* attrName, char** str) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    WARN("Attribute %s of node %s not found\n", attrName, node->name);
    return ncclInternalError;
  }
  if (node->attrs[index].type != KEY_TYPE_STR) {
    WARN("Attribute %s of node %s is not a string (%d)\n", attrName, node->name, node->attrs[index].type);
    return ncclInternalError;
  }
  *str = node->attrs[index].strValue;
  return ncclSuccess;
}

static ncclResult_t xmlFindTag(struct xmlSystem* system, const char* tagName, struct xmlNode** node) {
  *node = NULL;
  for (int i=0; i<system->maxIndex; i++) {
    struct xmlNode* n = system->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlFindTagKvStr(struct xmlSystem* system, const char* tagName, struct xmlNode** node, const char* attrName, const char* attrValue) {
  *node = NULL;
  for (int i=0; i<system->maxIndex; i++) {
    struct xmlNode* n = system->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      int index;
      NCCLCHECK(xmlGetAttrIndex(n, attrName, &index));
      if (index != -1) {
        char* strValue;
        NCCLCHECK(xmlGetAttrStr(n, attrName, &strValue));
        if (strcmp(strValue, attrValue) == 0) {
          *node = n;
          return ncclSuccess;
        }
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlSetAttrInt(struct xmlNode* node, const char* attrName, int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
  }
  node->attrs[index].type = KEY_TYPE_INT;
  node->attrs[index].intValue = value;
  return ncclSuccess;
}
static ncclResult_t xmlSetAttrStr(struct xmlNode* node, const char* attrName, const char* str) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
  }
  node->attrs[index].type = KEY_TYPE_STR;
  strncpy(node->attrs[index].strValue, str, MAX_STR_LEN);
  return ncclSuccess;
}

static ncclResult_t xmlGetSub(struct xmlNode* node, const char* subName, struct xmlNode** sub) {
  *sub = NULL;
  for (int s=0; s<node->nSubs; s++) {
    if (strcmp(node->subs[s]->name, subName) == 0) {
      *sub = node->subs[s];
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlGetSubKvStr(struct xmlNode* node, const char* subName, struct xmlNode** sub, const char* attrName, const char* attrValue) {
  *sub = NULL;
  for (int s=0; s<node->nSubs; s++) {
    struct xmlNode* subNode = node->subs[s];
    if (strcmp(subNode->name, subName) == 0) {
      int index;
      NCCLCHECK(xmlGetAttrIndex(subNode, attrName, &index));
      if (index != -1) {
        char* strValue;
        NCCLCHECK(xmlGetAttrStr(subNode, attrName, &strValue));
        if (strcmp(strValue, attrValue) == 0) {
          *sub = node->subs[s];
          return ncclSuccess;
        }
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlGetSubKvInt(struct xmlNode* node, const char* subName, struct xmlNode** sub, const char* attrName, int attrValue) {
  *sub = NULL;
  for (int s=0; s<node->nSubs; s++) {
    struct xmlNode* subNode = node->subs[s];
    if (strcmp(subNode->name, subName) == 0) {
      int index;
      NCCLCHECK(xmlGetAttrIndex(subNode, attrName, &index));
      if (index != -1) {
        int intValue;
        NCCLCHECK(xmlGetAttrInt(subNode, attrName, &intValue));
        if (intValue == attrValue) {
          *sub = node->subs[s];
          return ncclSuccess;
        }
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlAddSub(struct xmlSystem* system, struct xmlNode* node, const char* subName, struct xmlNode** sub) {
  struct xmlNode* s = system->nodes+system->maxIndex++;
  *sub = s;
  s->parent = node;
  node->subs[node->nSubs++] = s;
  strncpy(s->name, subName, MAX_STR_LEN);
  return ncclSuccess;
}

#endif
