/* OpenCL runtime library: file & directory related utility functions

   Copyright (c) 2016 Romaric JODIN
   Copyright (c) 2016-2021 Michal Babej / Tampere University
   Copyright (c) 2019 Kati Tervo / Tampere University
   Copyright (c) 2024 Michal Babej / Intel Finland Oy
   Copyright (c) 2024 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#ifndef _WIN32
#define _GNU_SOURCE
#define _DEFAULT_SOURCE
#define _BSD_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <libgen.h>
#include <unistd.h>
#else
#include "vccompat.hpp"
#ifdef __MINGW32__
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "pocl.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_util.h"

#ifdef __ANDROID__

int pocl_mkstemp(char *path);

#endif

/*****************************************************************************/

int
pocl_rm_rf(const char* path) 
{
  DIR *d = opendir(path);
  size_t path_len = strlen(path);
  int error = -1;
  
  if(d) 
    {
      struct dirent *p = readdir(d);
      error = 0;
      while (!error && p)
        {
          char *buf;
          if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, ".."))
            {
              p = readdir (d);
              continue;
            }

          size_t len = path_len + strlen(p->d_name) + 2;
          buf = malloc(len);
          if (buf)
            {
              struct stat statbuf;
              snprintf(buf, len, "%s/%s", path, p->d_name);

              if (lstat (buf, &statbuf) < 0)
                {
                  POCL_MSG_ERR ("Can't get lstat() on %s\n", buf);
                  closedir (d);
                  free (buf);
                  return -1;
                }
              if (S_ISDIR (statbuf.st_mode) && !S_ISLNK (statbuf.st_mode))
                error = pocl_rm_rf (buf);
              else
                error = remove (buf);

              free(buf);
            }
          else
            {
              closedir (d);
              POCL_MSG_ERR ("out of memory");
              return -1;
            }

          p = readdir(d);
        }
      closedir(d);
      
      if (!error)
        error = remove (path);
    }
  return error;
}


int
pocl_mkdir_p (const char* path)
{
  size_t len = strlen (path);
  if (len >= POCL_MAX_PATHNAME_LENGTH - 1)
    {
      return -1;
    }
  if (len <= 1)
    return -1;

  char path_copy[POCL_MAX_PATHNAME_LENGTH];
  memcpy (path_copy, path, len);
  path_copy[len] = 0;

  for (char *tmp = path_copy + 1; *tmp; tmp++)
    {
      if (*tmp == '/')
        {
          *tmp = '\0';
          errno = 0;
          if (mkdir (path_copy, S_IRWXU) != 0)
            {
              if (errno != EEXIST)
                return -1;
            }
          *tmp = '/';
        }
    }

  if (mkdir (path_copy, S_IRWXU) != 0)
    {
      if (errno != EEXIST)
        return -1;
    }

  return 0;
}

int
pocl_remove(const char* path) 
{
  return remove(path);
}

int
pocl_exists(const char* path) 
{
  return !access(path, R_OK);
}

int 
pocl_touch_file(const char* path) 
{
  FILE *f = fopen(path, "w");
  if (f)
    {
      fclose(f);
      return 0;
    }
  return -1;
}
/****************************************************************************/

#define CHUNK_SIZE (2 * 1024 * 1024)

int
pocl_read_file(const char* path, char** content, uint64_t *filesize) 
{
  assert(content);
  assert(path);
  assert(filesize);
  *content = NULL;
  *filesize = 0;

  /* files in /proc return zero size, while
     files in /sys return size larger than actual actual content size;
     this reads the content sequentially. */
  size_t total_size = 0;
  size_t actually_read = 0;
  char *ptr = (char *)malloc (CHUNK_SIZE + 1);
  if (ptr == NULL)
    return -1;

  FILE *f = fopen (path, "rb");
  if (f == NULL) {
    POCL_MSG_ERR ("fopen( %s ) failed\n", path);
    goto ERROR;
  }

  do
    {
      char *reallocated = (char *)realloc (ptr, (total_size + CHUNK_SIZE + 1));
      if (reallocated == NULL)
        {
          fclose (f);
          goto ERROR;
        }
      ptr = reallocated;

      actually_read = fread (ptr + total_size, 1, CHUNK_SIZE, f);
      total_size += actually_read;
    }
  while (actually_read == CHUNK_SIZE);

  if (ferror (f))
  {
    fclose (f);
    goto ERROR;
  }

  if (fclose (f))
    goto ERROR;

  /* add an extra NULL character for strings */
  ptr[total_size] = 0;
  *content = ptr;
  *filesize = (uint64_t)total_size;
  return 0;

ERROR:
  free (ptr);
  return -1;
}

/* Atomic write - with rename() */
int
pocl_write_file (const char *path, const char *content, uint64_t count,
                 int append)
{
  assert(path);
  assert(content);
  char path2[POCL_MAX_PATHNAME_LENGTH];
  int err, fd = -1;

  if (append)
    {
      fd = open (path, O_RDWR | O_APPEND | O_CREAT, 0660);
      err = fd < 0;
    }
  else
    {
      err = pocl_mk_tempname (path2, path, ".temp", &fd);
    }

  if (err)
    {
      POCL_MSG_ERR ("open(%s) failed\n", path);
      return -1;
    }

  ssize_t res = write (fd, content, (size_t)count);
  if (res < 0 || (size_t)res < (size_t)count)
    {
      POCL_MSG_ERR ("write(%s) failed\n", path);
      close (fd);
      return -1;
    }

#ifdef HAVE_FDATASYNC
  if (fdatasync (fd))
    {
      POCL_MSG_ERR ("fdatasync() failed\n");
      close (fd);
      return errno;
    }
#elif defined(HAVE_FSYNC)
  if (fsync (fd))
    {
      POCL_MSG_ERR ("fsync() failed\n");
      close (fd);
      return errno;
    }
#endif

  if (close (fd) < 0)
    return errno;

  if (append)
    return 0;
  else
    return pocl_rename (path2, path);
}

/****************************************************************************/

int pocl_rename(const char *oldpath, const char *newpath) {
  return rename (oldpath, newpath);
}

int
pocl_mk_tempname (char *output, const char *prefix, const char *suffix,
                  int *ret_fd)
{
#if defined(_WIN32)
  char buf[256];
  int ok = GetTempFileName(getenv("TEMP"), prefix, 0, buf);
  return ok ? 0 : 1;
#elif defined(HAVE_MKOSTEMPS) || defined(HAVE_MKSTEMPS) || defined(__ANDROID__)
  /* using mkstemp() instead of tmpnam() has no real benefit
   * here, as we have to pass the filename to llvm,
   * but tmpnam() generates an annoying warning... */
  int fd;

  strncpy (output, prefix, POCL_MAX_PATHNAME_LENGTH);
  size_t len = strlen (prefix);
  strncpy (output + len, "_XXXXXX", (POCL_MAX_PATHNAME_LENGTH - len));

#ifdef __ANDROID__
  fd = pocl_mkstemp (output);
#else
  if (suffix)
    {
      len += 7;
      strncpy (output + len, suffix, (POCL_MAX_PATHNAME_LENGTH - len));
#ifdef HAVE_MKOSTEMPS
      fd = mkostemps (output, strlen (suffix), O_CLOEXEC);
#else
      fd = mkstemps (output, strlen (suffix));
#endif
    }
  else
#ifdef HAVE_MKOSTEMPS
    fd = mkostemp (output, O_CLOEXEC);
#else
    fd = mkstemp (output);
#endif
#endif

  if (fd < 0)
    {
      POCL_MSG_ERR ("mkstemp() failed\n");
      return errno;
    }

  int err = 0;
  if (ret_fd)
    *ret_fd = fd;
  else
    err = close (fd);

  return err ? errno : 0;

#else
#error mkostemps() / mkstemps() both unavailable
#endif
}

int
pocl_mk_tempdir (char *output, const char *prefix)
{
#if defined(_WIN32)
  assert (0);
#elif defined(HAVE_MKDTEMP)
  /* TODO mkdtemp() might not be portable outside Linux */
  strncpy (output, prefix, POCL_MAX_PATHNAME_LENGTH);
  size_t len = strlen (prefix);
  strncpy (output + len, "_XXXXXX", (POCL_MAX_PATHNAME_LENGTH - len));
  return (mkdtemp (output) == NULL);
#else
#error mkdtemp() not available
#endif
}

int
pocl_write_tempfile (char *output_path,
                     const char *prefix,
                     const char *suffix,
                     const char *content,
                     uint64_t count)
{
  assert (output_path);
  assert (prefix);
  assert (suffix);
  assert (content);

  int fd = -1, err = 0;

  err = pocl_mk_tempname (output_path, prefix, suffix, &fd);
  if (err)
    {
      POCL_MSG_ERR ("pocl_mk_tempname() failed\n");
      return err;
    }

  size_t bytes = count;
  ssize_t res;
  while (bytes > 0)
    {
      res = write (fd, content, bytes);
      if (res < 0)
        {
          POCL_MSG_ERR ("write(%s) failed\n", output_path);
          return errno;
        }
      else
        {
          bytes -= res;
          content += res;
        }
    }

#ifdef HAVE_FDATASYNC
  if (fdatasync (fd))
    {
      POCL_MSG_ERR ("fdatasync() failed\n");
      return errno;
    }
#elif defined(HAVE_FSYNC)
  if (fsync (fd))
    return errno;
#endif

  err = close (fd);

  return err ? -2 : 0;
}

char *
pocl_parent_path (char *path)
{
  return strcpy(path, dirname (path));
}

pocl_file_type
pocl_get_file_type (const char *path)
{
  struct stat st;
  if (stat (path, &st) != 0)
    return POCL_FS_STATUS_ERROR;

  if (S_ISREG (st.st_mode))
    return POCL_FS_REGULAR;

  if (S_ISDIR (st.st_mode))
    return POCL_FS_DIRECTORY;

  assert (!"TODO: mapping of a non-file/-directory.");
  return POCL_FS_STATUS_ERROR;
}

typedef struct dirent_handle_s
{
  DIR *dir;
  struct dirent *entry;
  const char *basedir; /* The path given in pocl_dir_iterator(). */
  const char *path;    /* <basedir>/<entry->d_name> */
} dirent_handle;

int
pocl_dir_iterator (const char *path, pocl_dir_iter *iter)
{
  /* dirent_handle variants: iter->handle == NULL
   * || ((dirent_handle *)iter->handle)->dir != NULL. */

  dirent_handle *handle_impl = calloc (1, sizeof (dirent_handle));
  if (handle_impl == NULL)
    {
      return -1;
    }

  DIR *d = opendir (path);
  if (d == NULL)
    {
      free (handle_impl);
      iter->handle = NULL;
      return -1;
    }


  handle_impl->dir = d;
  handle_impl->basedir = path;
  iter->handle = handle_impl;
  return 0;
}

int
pocl_dir_next_entry (pocl_dir_iter iter)
{
  assert (iter.handle != NULL && "Must call pocl_dir_iterator() first!");
  dirent_handle *handle_impl = iter.handle;
  assert (handle_impl->dir != NULL && "Broken invariant!");

  while ((handle_impl->entry = readdir (handle_impl->dir)))
    {
      if (strcmp (handle_impl->entry->d_name, ".") == 0)
        continue;
      if (strcmp (handle_impl->entry->d_name, "..") == 0)
        continue;
      return 1;
    }

  return 0;
}

const char *
pocl_dir_iter_get_path (pocl_dir_iter iter)
{
  assert (iter.handle != NULL && "Must call pocl_dir_iterator() first!");
  dirent_handle *handle_impl = iter.handle;
  assert (handle_impl->entry && "Must call pocl_dir_next_entry() first!");

  if (handle_impl->path)
    free ((void *)handle_impl->path);
  const char *full_path[]
      = { handle_impl->basedir, "/", handle_impl->entry->d_name };
  handle_impl->path = pocl_strcatdup_v (3, full_path);
  return handle_impl->path;
}

void
pocl_release_dir_iterator (pocl_dir_iter *iter)
{
  assert (iter && "Invalid pocl_dir_iter handle!");

  if (iter->handle == NULL)
    return;

  dirent_handle *handle_impl = iter->handle;
  assert (handle_impl->dir != NULL && "Broken invariant!");
  closedir (handle_impl->dir);
  free ((void *)handle_impl->path);
  free (handle_impl);
  iter->handle = NULL;
}
